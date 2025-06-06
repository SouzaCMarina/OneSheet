# -- coding: utf-8 --
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware import Middleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
import io
import numpy as np
import unicodedata
import re
from typing import Optional

# =============================================
# Configurações da API
# =============================================
app = FastAPI(
    title="API de Processamento de Planilhas",
    description="Processa planilhas financeiras e retorna dados tratados",
    middleware=[Middleware(GZipMiddleware)]
)

# Limites e configurações
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
ALLOWED_TYPES = [
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel"
]

# =============================================
# Middleware de Limite de Tamanho
# =============================================
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == 'POST' and 'content-length' in request.headers:
        content_length = int(request.headers['content-length'])
        if content_length > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo muito grande. Tamanho máximo permitido: {MAX_FILE_SIZE//(1024*1024)}MB"
            )
    return await call_next(request)

# =============================================
# Funções Auxiliares (Mantidas do Original)
# =============================================
def remover_acentos(texto: str) -> str:
    """Remove acentos e caracteres especiais de um texto"""
    nfkd = unicodedata.normalize('NFKD', texto)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])

def normalizar_cfop(cfop: str) -> str:
    """Normaliza códigos CFOP para comparação"""
    return str(cfop).strip().lower().replace('.0', '')

def extrair_numero_texto(texto: str) -> Optional[str]:
    """Extrai números de um texto para conciliação"""
    match = re.search(r'\d+', str(texto))
    return match.group(0) if match else None

# =============================================
# Dicionário de Classificação CFOP
# =============================================
CLASSIFICACAO_CFOP = {
    # Vendas
    '5101': 'Vendas', '5102': 'Vendas', '6101': 'Vendas', '7101': 'Vendas',
    '7102': 'Vendas', '7127': 'Vendas',
    
    # Transferências
    '5151': 'Transferências', '5557': 'Transferências', '5152': 'Transferências',
    '5209': 'Transferências', '5552': 'Transferências', '1151': 'Transferências',
    '1152': 'Transferências', '1557': 'Transferências',
    
    # Remessas
    '5915': 'Remessas', '5920': 'Remessas', '7949': 'Remessas',
    '5949': 'Remessas', '5909': 'Remessas', '5927': 'Remessas',
    
    # ... (adicione todas as outras entradas do seu dicionário original)
}

# =============================================
# Função Principal de Processamento
# =============================================
def processar_dados(all_sheets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processa as planilhas conforme a lógica original"""
    # 1. Processa a sheet "Razão Contábil"
    try:
        df_razao = all_sheets['Razão Contábil']
    except KeyError:
        possible_names = ['razao contabil', 'Razao Contabil', 'Razão', 'razao']
        for name in possible_names:
            if name.lower() in [s.lower() for s in all_sheets.keys()]:
                df_razao = all_sheets[name]
                break
        else:
            raise ValueError("Sheet 'Razão Contábil' não encontrada")

    # Padroniza colunas
    df_razao.columns = [remover_acentos(col.strip().lower()).replace(' ', '_') for col in df_razao.columns]

    # Classificação dos lançamentos
    def classificar_lancamentos(row):
        doc_type = str(row['document_type']).strip().upper()
        if doc_type == 'CA':
            return 'Importação'
        elif doc_type == 'RE':
            return 'Entradas'
        elif doc_type in ['AB', 'SA']:
            return 'Manuais'
        elif doc_type == 'SL':
            return 'Exclusão'
        elif doc_type == 'SX':
            return 'GAAP'
        else:
            return row['classificacao'] if pd.notna(row['classificacao']) else 'Outros'

    df_razao['classificacao'] = df_razao.apply(classificar_lancamentos, axis=1)
    df_razao['origem'] = 'razao'

    # Seleção de colunas
    df_razao_tratado = df_razao[
        ['classificacao', 'business_place', 'user_name', 'document_number', 'reference',
         'amount_in_functional_currency', 'text', 'transaction_code', 'tax_code',
         'posting_date', 'document_date', 'document_type', 'g/l_account', 'origem']
    ]

    # 2. Processa a sheet "Saídas"
    try:
        df_saida = all_sheets['Saídas']
    except KeyError:
        possible_names = ['saidas', 'Saidas', 'SAIDAS', 'SAÍDAS']
        for name in possible_names:
            if name.lower() in [s.lower() for s in all_sheets.keys()]:
                df_saida = all_sheets[name]
                break
        else:
            raise ValueError("Sheet 'Saídas' não encontrada")

    df_saida.columns = [remover_acentos(col.strip().lower()).replace(' ', '_') for col in df_saida.columns]
    df_saida = df_saida.dropna(how='all')
    df_saida['classificacao'] = df_saida['cfop'].apply(
        lambda x: CLASSIFICACAO_CFOP.get(normalizar_cfop(x), 'Não classificado')
    )

    # Mapeamento de colunas
    df_saida['user_name'] = df_saida['nf_created_by']
    df_saida['document_number'] = df_saida['doc.number']
    df_saida['reference'] = df_saida['nf']
    df_saida['amount_in_functional_currency'] = df_saida['pis_tax_value']
    df_saida['text'] = df_saida.get('observ', '')
    df_saida['transaction_code'] = ''
    df_saida['tax_code'] = ''
    df_saida['posting_date'] = df_saida['post._date']
    df_saida['document_date'] = ''
    df_saida['document_type'] = df_saida['cfop']
    df_saida['g/l_account'] = 'fiscal'
    df_saida['origem'] = 'saida'

    df_saida_tratada = df_saida[
        ['classificacao', 'business_place', 'user_name', 'document_number', 'reference',
         'amount_in_functional_currency', 'text', 'transaction_code', 'tax_code',
         'posting_date', 'document_date', 'document_type', 'g/l_account', 'origem']
    ]

    # 3. Processa a sheet "Entradas"
    try:
        df_entrada = all_sheets['Entradas']
    except KeyError:
        possible_names = ['entradas', 'Entradas', 'ENTRADAS', 'entrada']
        for name in possible_names:
            if name.lower() in [s.lower() for s in all_sheets.keys()]:
                df_entrada = all_sheets[name]
                break
        else:
            raise ValueError("Sheet 'Entradas' não encontrada")

    df_entrada.columns = [remover_acentos(col.strip().lower()).replace(' ', '_') for col in df_entrada.columns]
    df_entrada = df_entrada.rename(columns={'classificacao_reconciliacao': 'classificacao'})
    df_entrada = df_entrada.dropna(how='all')
    df_entrada['classificacao'] = df_entrada['cfop'].apply(
        lambda x: CLASSIFICACAO_CFOP.get(normalizar_cfop(x), 'Não classificado')
    )

    # Mapeamento de colunas
    df_entrada['user_name'] = df_entrada['nf_created_by']
    df_entrada['document_number'] = df_entrada['doc.number']
    df_entrada['reference'] = df_entrada['nf']
    df_entrada['amount_in_functional_currency'] = df_entrada['pis_tax_value']
    df_entrada['text'] = df_entrada.get('observ', '')
    df_entrada['transaction_code'] = ''
    df_entrada['tax_code'] = ''
    df_entrada['posting_date'] = df_entrada['post._date']
    df_entrada['document_date'] = ''
    df_entrada['document_type'] = df_entrada['cfop']
    df_entrada['g/l_account'] = 'fiscal'
    df_entrada['origem'] = 'entrada'

    df_entrada_tratada = df_entrada[
        ['classificacao', 'business_place', 'user_name', 'document_number', 'reference',
         'amount_in_functional_currency', 'text', 'transaction_code', 'tax_code',
         'posting_date', 'document_date', 'document_type', 'g/l_account', 'origem']
    ]

    # 4. Unificação e conciliação
    df_unificado = pd.concat([df_razao_tratado, df_entrada_tratada, df_saida_tratada], ignore_index=True)

    # Processamento adicional
    df_unificado['reference'] = df_unificado['reference'].astype(str)
    df_unificado[['reference', 'serie']] = df_unificado['reference'].str.split('-', n=1, expand=True)
    df_unificado['reference'] = df_unificado['reference'].str.strip()
    df_unificado['serie'] = df_unificado['serie'].str.strip().fillna('')

    # Inversão de valores para CFOPs de entrada
    cfops_entrada = [cfop for cfop, classificacao in CLASSIFICACAO_CFOP.items() if classificacao == 'Entradas']
    df_unificado['document_type'] = df_unificado['document_type'].astype(str)
    df_unificado['amount_in_functional_currency'] = pd.to_numeric(
        df_unificado['amount_in_functional_currency'], errors='coerce'
    )
    df_unificado.loc[
        df_unificado['document_type'].isin(cfops_entrada),
        'amount_in_functional_currency'
    ] *= -1

    # Extração de número do texto (para conciliação)
    mascara_razao_com_texto = (
        (df_unificado['origem'] == 'razao') & 
        (df_unificado['text'].notna()) & 
        (df_unificado['text'].str.contains(r'\d', regex=True))
    )
    df_unificado.loc[mascara_razao_com_texto, 'reference'] = df_unificado.loc[
        mascara_razao_com_texto, 'text'
    ].apply(extrair_numero_texto)

    # Classificação de lojinha
    mascara_loja = (
        (df_unificado['classificacao'] == 'Vendas') &
        (df_unificado['serie'].isin(['2', '4'])) &
        (
            (df_unificado['reference'] == '0') |
            (df_unificado['reference'].str.lower().str.contains('ajuste lojinha', na=False))
        )
    )
    df_unificado.loc[mascara_loja, 'classificacao'] = 'Lojinha'

    # Ordenação
    df_unificado = df_unificado.sort_values(by=['reference', 'origem'], ignore_index=True)

    # Cálculo da conciliação
    soma_por_nota = df_unificado.groupby('document_number')['amount_in_functional_currency'].transform('sum')
    df_unificado['conciliacao'] = np.where(np.isclose(soma_por_nota, 0), 'OK', 'Pendência')

    # Separação final
    df_ok = df_unificado[df_unificado['conciliacao'] == 'OK']
    df_pendencia = df_unificado[df_unificado['conciliacao'] == 'Pendência']

    return df_ok, df_pendencia

# =============================================
# Endpoint Principal
# =============================================
@app.post("/processar")
async def processar_planilha(arquivo: UploadFile = File(...)):
    try:
        # 1. Validações iniciais
        if arquivo.content_type not in ALLOWED_TYPES:
            raise HTTPException(400, "Formato inválido. Envie um arquivo Excel (.xlsx ou .xls)")
        
        # 2. Leitura controlada
        file_bytes = await arquivo.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(413, f"Tamanho excede 200MB (tamanho atual: {len(file_bytes)//(1024*1024)}MB)")
        
        # 3. Processamento em memória
        with io.BytesIO(file_bytes) as file_stream:
            all_sheets = pd.read_excel(file_stream, sheet_name=None)
            
            # Processa os dados
            df_ok, df_pendencia = processar_dados(all_sheets)
            
            # Cria o Excel em memória
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_ok.to_excel(writer, sheet_name="OK", index=False)
                df_pendencia.to_excel(writer, sheet_name="Pendências", index=False)
            
            output.seek(0)
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment; filename=planilha_processada.xlsx"}
            )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(500, f"Erro interno: {str(e)}")

# =============================================
# Health Check
# =============================================
@app.get("/")
async def health_check():
    return {"status": "online", "message": "API de processamento de planilhas"}