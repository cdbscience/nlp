from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import default_storage
from .models import Document, Query, File
from .text_processing import text_processing
import os
import json
import sys
import traceback

# Tentar importar numpy
try:
    import numpy as np
except ImportError:
    np = None

from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Tentar importar LangChain com fallback
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Qdrant
    import pandas as pd
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"AVISO: LangChain não está instalado: {e}")
    LANGCHAIN_AVAILABLE = False


def cosine_similarity(vec1, vec2):
    """Calcula similaridade de cosseno entre dois vetores"""
    if np is None:
        raise ImportError("numpy é necessário. Instale com: pip install numpy")
    
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return float(dot_product / (norm_vec1 * norm_vec2))
    except Exception as e:
        print(f"Erro ao calcular similaridade: {e}")
        return 0.0


def extract_pdf_text(file_path):
    """Extrai texto de um arquivo PDF usando LangChain"""
    if not LANGCHAIN_AVAILABLE:
        # Fallback com pdfplumber
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber é necessário. Instale com: pip install pdfplumber")
        
        try:
            text_content = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_content.append(f"--- Página {page_num} ---\n{text}")
            return "\n\n".join(text_content)
        except Exception as e:
            raise Exception(f"Erro ao extrair PDF com pdfplumber: {str(e)}")
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_content = []
        for doc in documents:
            text_content.append(doc.page_content)
        
        return "\n\n".join(text_content)
    except Exception as e:
        raise Exception(f"Erro ao extrair PDF com LangChain: {str(e)}")

def get_embeddings():
    """Retorna instância do OpenAI Embeddings via LangChain"""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain não está instalado. Instale com: pip install -r rag_requirements.txt")
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small"
        )
        return embeddings
    except Exception as e:
        raise Exception(f"Erro ao inicializar embeddings: {str(e)}")


def get_openai_client():
    """Retorna cliente OpenAI"""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada")
        return OpenAI(api_key=api_key)
    except Exception as e:
        raise Exception(f"Erro ao inicializar OpenAI client: {str(e)}")


def generate_rag_answer(question, context_docs):
    """Gera resposta usando OpenAI baseada nos documentos relevantes"""
    try:
        client = get_openai_client()
        
        # Preparar contexto
        context_text = "\n\n".join([
            f"Documento {i+1}: {doc['title']}\n{doc['content']}"
            for i, doc in enumerate(context_docs)
        ])
        
        prompt = f"""Você é um assistente inteligente que responde perguntas baseado em documentos fornecidos.

Documentos relevantes:
{context_text}

Pergunta: {question}

Instruções:
- Responda de forma clara e objetiva
- Baseie sua resposta apenas nas informações dos documentos fornecidos
- Se a informação não estiver nos documentos, diga que não encontrou a resposta
- Mantenha a resposta concisa mas completa
- Cite o documento de origem quando relevante

Resposta:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente útil que responde baseado em documentos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Erro ao gerar resposta com OpenAI: {str(e)}")
        # Fallback: resposta baseada apenas nos documentos
        return f"Baseado nos documentos encontrados: {context_text[:500]}..."


class RAGManager:
    """Gerenciador do sistema RAG usando Qdrant"""
    
    def __init__(self):
        self.qdrant = None
        self.chat = None
        self.embed_model = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Inicializa os componentes do RAG"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            print(f"🔑 OPENAI_API_KEY presente: {bool(api_key)}")
            if not api_key:
                raise ValueError("OPENAI_API_KEY não encontrada")
            
            # Inicializar ChatOpenAI
            self.chat = ChatOpenAI(
                model='gpt-3.5-turbo',
                openai_api_key=api_key
            )
            
            # Inicializar embeddings
            self.embed_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=api_key
            )
            
            # Inicializar Qdrant em memória
            from langchain_core.documents import Document as LangChainDocument
            empty_doc = LangChainDocument(page_content="", metadata={})
            
            self.qdrant = Qdrant.from_documents(
                documents=[empty_doc],  # Começa com um documento vazio
                embedding=self.embed_model,
                location=":memory:",
                collection_name="rag_documents"
            )
            
            print("✅ RAG Manager inicializado com sucesso")
            
        except Exception as e:
            print(f"❌ Erro ao inicializar RAG Manager: {e}")
            import traceback
            print(traceback.format_exc())
            self.qdrant = None
            self.chat = None
            self.embed_model = None
    
    def add_documents(self, documents):
        """Adiciona documentos ao vector store"""
        if not self.qdrant:
            print("❌ Qdrant não inicializado")
            return False
        
        try:
            # Converter documentos Django para formato LangChain
            langchain_docs = []
            for doc in documents:
                from langchain_core.documents import Document as LangChainDocument
                langchain_doc = LangChainDocument(
                    page_content=doc.content,
                    metadata={
                        'id': doc.id,
                        'title': doc.title,
                        'file_name': doc.file.name,
                        'chunk_index': doc.chunk_index
                    }
                )
                langchain_docs.append(langchain_doc)
            
            # Adicionar ao Qdrant
            self.qdrant.add_documents(langchain_docs)
            print(f"✅ Adicionados {len(langchain_docs)} documentos ao Qdrant")
            
            # Verificar se os documentos foram adicionados
            total_docs = self.qdrant.client.count(collection_name="rag_documents")
            print(f"📊 Total de documentos no Qdrant: {total_docs}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao adicionar documentos: {e}")
            return False
    
    def custom_prompt(self, query: str, file_id: str = None):
        """Cria prompt personalizado baseado na busca por similaridade"""
        if not self.qdrant:
            return f"Erro: Sistema RAG não inicializado. Pergunta: {query}"
        
        try:
            # Buscar documentos mais similares
            if file_id and file_id != 'all':
                # Filtrar por arquivo específico usando metadata
                print(f"🔍 Filtrando por arquivo: {file_id}")
                filter_condition = {"file_name": file_id}
                results = self.qdrant.similarity_search(query, k=5, filter=filter_condition)
                print(f"📄 Encontrados {len(results)} documentos para o arquivo {file_id}")
            else:
                # Buscar em todos os documentos
                print("🔍 Buscando em todos os documentos")
                results = self.qdrant.similarity_search(query, k=5)
                print(f"📄 Encontrados {len(results)} documentos no total")
            
            if not results:
                return f"Não encontrei documentos relevantes para a pergunta: {query}"
            
            source_knowledge = "\n".join([x.page_content for x in results])
            
            augment_prompt = f"""Se baseie exclusivamente no contexto abaixo para responder à pergunta. Responda da forma mais humanizada possível.

Contexto:
{source_knowledge}

Pergunta: {query}"""
            return augment_prompt
            
        except Exception as e:
            print(f"Erro no custom_prompt: {e}")
            import traceback
            print(traceback.format_exc())
            return f"Erro ao processar pergunta: {query}"
    
    def get_response(self, query: str, file_id: str = None):
        """Gera resposta usando o sistema RAG"""
        print(f"🔍 Query: {query}, File: {file_id}")
        
        if not self.chat:
            return {
                "status": "error",
                "message": "Sistema de chat não inicializado",
                "query": query
            }
        
        try:
            prompt = self.custom_prompt(query, file_id)
            # print(f'Prompt gerado: {prompt}')
            
            response = self.chat.invoke(prompt)
            
            if hasattr(response, "content"):
                content = response.content
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)

            print(f"Resposta gerada: {content}")
            return {
                "status": "success",
                "response": content,
                "query": query,
            }
            
        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            return {
                "status": "error",
                "message": f"Erro ao processar pergunta: {str(e)}",
                "query": query
            }


# Instância global do RAG Manager
rag_manager = RAGManager()

def vectorize_text(text):
    """Vetoriza um texto usando OpenAI embeddings"""
    try:
        embeddings = get_embeddings()
        embedding_vector = embeddings.embed_query(text)
        return embedding_vector
    except Exception as e:
        # Registrar erro mas não falhar completamente
        print(f"AVISO - Erro ao vetorizar texto: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")
        # Retornar vetor vazio para não quebrar o processamento
        return []

def get_chunks(text, chunk_size=1000, overlap=200):
    """Divide o texto em chunks com tamanho e overlap especificados"""
    if not text:
        return []
    
    normalized = text_processing(text)
    
    words = normalized
    chunks = []
    start = 0
    text_length = len(words)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        if end == text_length:
            break
        
        start += chunk_size - overlap
    
    return chunks

def process_pdf_file(uploaded_file):
    """Processa arquivo PDF: extrai texto, cria chunks, vetoriza e salva no banco
    
    Args:
        uploaded_file: Um arquivo Django UploadedFile ou um objeto similiar com name, size
        
    Returns:
        dict: {'status': 'success'/'error', 'file_id', 'file_name', 'document_count', ...}
    """
    try:
        try:
            # Salvar arquivo temporariamente
            temp_path = default_storage.save(f'temp/{uploaded_file.name}', uploaded_file)
            full_path = default_storage.path(temp_path)
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Erro ao salvar arquivo: {str(e)}"
            }
        
        try:
            # Extrair texto usando LangChain
            pdf_text = extract_pdf_text(full_path)
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Erro ao extrair texto do PDF: {str(e)}"
            }
        
        try:
            # Dividir em chunks usando LangChain
            chunks = get_chunks(pdf_text, chunk_size=1000, overlap=200)
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Erro ao dividir texto em chunks: {str(e)}"
            }
        
        if not chunks:
            return {
                'status': 'error',
                'error': "Nenhum chunk foi criado. O PDF pode estar vazio."
            }
        
        try:
            # Criar ou obter File
            file_obj = File.objects.create(
                name=uploaded_file.name,
                file_path=temp_path,
                file_size=uploaded_file.size,
                total_documents=len(chunks),
                is_processed=False  # Será True após processar todos os chunks
            )
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Erro ao criar registro de arquivo: {str(e)}"
            }
        
        # Criar Documents para cada chunk com embeddings
        document_ids = []
        processing_errors = []
        
        for index, chunk in enumerate(chunks):
            try:
                # Vetorizar o chunk
                embedding = vectorize_text(chunk)
                
                # Criar documento com embedding
                doc = Document.objects.create(
                    file=file_obj,
                    title=f"{uploaded_file.name} - Parte {index + 1}",
                    content=chunk,
                    chunk_index=index,
                    word_count=len(chunk.split()),
                    char_count=len(chunk),
                    metadata={
                        'embedding_model': 'text-embedding-3-small',
                        'source_file': uploaded_file.name,
                    }
                )
                
                # Salvar embedding como JSON string
                doc.set_embedding(embedding)
                doc.save()
                
                document_ids.append(doc.id)
                
            except Exception as e:
                error_msg = f"Erro ao vetorizar chunk {index}: {str(e)}"
                print(error_msg)
                processing_errors.append(error_msg)
                continue
        
        # Marcar arquivo como processado
        file_obj.is_processed = True
        file_obj.total_documents = len(document_ids)
        file_obj.save()
        
        if not document_ids:
            return {
                'status': 'error',
                'error': f"Nenhum documento foi criado. Erros: {'; '.join(processing_errors)}"
            }
        
        # Adicionar documentos ao RAG Manager (Qdrant)
        created_documents = Document.objects.filter(id__in=document_ids)
        rag_manager.add_documents(created_documents)
        
        print(f'Arquivos processados: {len(document_ids)} documentos criados para o arquivo {uploaded_file.name}')
        return {
            'status': 'success',
            'file_id': file_obj.id,
            'file_name': file_obj.name,
            'document_count': len(document_ids),
            'document_ids': document_ids
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Erro geral no processamento: {error_detail}")
        return {
            'status': 'error',
            'error': f"Erro ao processar PDF: {str(e)}"
        }

def handle_pdf_upload(request):
    """View handler para upload de PDF via requisição POST
    
    Retorna JsonResponse com resultado do processamento
    """
    if request.method == 'POST' and 'pdf_file' in request.FILES:
        # Verificar permissão de admin
        if not request.user.is_staff and not request.user.is_superuser:
            return JsonResponse({
                'status': 'error',
                'error': 'Only administrators can upload files'
            }, status=403)
        
        pdf_file = request.FILES['pdf_file']
        result = process_pdf_file(pdf_file)
        
        if result.get('status') == 'success':
            return JsonResponse(result, status=200)
        else:
            return JsonResponse(result, status=400)
    
    return JsonResponse({
        'status': 'error',
        'error': 'Invalid request'
    }, status=400)

@csrf_exempt
def index(request):
    """Main RAG interface"""
    if request.method == 'POST':
        # Verificar se é upload de PDF
        if 'pdf_file' in request.FILES:
            # Verificar permissão de admin
            if not request.user.is_staff and not request.user.is_superuser:
                print(f"[ERRO] Usuário {request.user.username} tentou fazer upload sem permissão")
                return JsonResponse({
                    'status': 'error',
                    'error': 'Only administrators can upload files'
                }, status=403)
            
            try:
                print(f"\n{'='*60}")
                print(f"[LOG] Iniciando upload de PDF")
                print(f"[LOG] Usuário: {request.user.username}")
                print(f"[LOG] LangChain Available: {LANGCHAIN_AVAILABLE}")
                print(f"[LOG] NumPy Available: {np is not None}")
                
                uploaded_file = request.FILES['pdf_file']
                print(f"[LOG] Arquivo: {uploaded_file.name}, Tamanho: {uploaded_file.size} bytes")
                
                # Validar se é PDF
                if not uploaded_file.name.endswith('.pdf'):
                    print(f"[ERRO] Formato inválido: {uploaded_file.name}")
                    return JsonResponse({
                        'status': 'error',
                        'error': 'Apenas arquivos PDF são aceitos'
                    }, status=400)
                
                # Processar PDF
                print(f"[LOG] Processando PDF...")
                result = process_pdf_file(uploaded_file)
                print(f"[LOG] Resultado: {result['status']}")
                print(f"{'='*60}\n")
                if result['status'] == 'success':
                    return JsonResponse(result)
                else:
                    return JsonResponse(result, status=400)
            
            except Exception as e:
                import traceback
                print(f"Erro no upload: {traceback.format_exc()}")
                return JsonResponse({
                    'status': 'error',
                    'error': f"Erro ao processar upload: {str(e)}"
                }, status=500)
        
        # Processar query de pergunta
        question = request.POST.get('question', '')
        selected_file_id = request.POST.get('file_id', 'all')
        
        if question:
            try:
                # Determinar o nome do arquivo para exibição
                if selected_file_id and selected_file_id != 'all':
                    try:
                        selected_file = File.objects.get(id=int(selected_file_id))
                        file_name = selected_file.name
                        file_id_for_rag = selected_file.name  # Passar o nome do arquivo para o RAG
                    except (ValueError, File.DoesNotExist):
                        file_name = "Arquivo não encontrado"
                        file_id_for_rag = None
                else:
                    file_name = "todos os arquivos"
                    file_id_for_rag = None
                
                # Usar o RAG Manager para gerar resposta
                rag_manager.add_documents(Document.objects.all())  # Garantir que todos os documentos estão carregados
                response_data = rag_manager.get_response(question, file_id_for_rag)
                
                # Salvar a query no banco
                query = Query.objects.create(question=question)
                query.answer = response_data.get('response', 'Erro na resposta')
                query.save()
                
                # Retornar resposta formatada
                if response_data['status'] == 'success':
                    return JsonResponse({
                        'question': question,
                        'answer': response_data['response'],
                        'file_name': file_name
                    })
                else:
                    return JsonResponse({
                        'question': question,
                        'answer': f"Erro: {response_data['message']}",
                        'file_name': file_name
                    })
                
            except Exception as e:
                print(f"Erro ao processar query: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return JsonResponse({
                    'question': question,
                    'answer': f"Erro ao processar pergunta: {str(e)}",
                    'file_name': 'Sistema RAG'
                })
    
    files = File.objects.all()
    documents = Document.objects.all()[:50]  # Últimos 50 documentos
    context = {
        'files': files,
        'documents': documents
    }
    return render(request, 'rag.html', context)

@csrf_exempt
def query_api(request):
    """API endpoint for RAG queries"""
    if request.method == 'POST':
        question = request.POST.get('question', '')
        
        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)
        
        try:
            query = Query.objects.create(question=question)
            
            # TODO: Implement RAG logic here
            answer = "Resposta do sistema RAG"
            
            query.answer = answer
            query.save()
            
            return JsonResponse({
                'id': query.id,
                'question': question,
                'answer': answer
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
