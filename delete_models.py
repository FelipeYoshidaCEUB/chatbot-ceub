"""
Script de gerenciamento de modelos do HuggingFace.

Este script permite listar e excluir modelos do cache do HuggingFace,
ajudando a liberar espaço em disco quando necessário.
"""

import os
import shutil

def get_directory_size(path):
    """
    Calcula o tamanho total de um diretório em bytes.
    
    Args:
        path: Caminho do diretório a ser analisado
    
    Returns:
        Tamanho total em bytes (0 se o diretório não existir)
    """
    total_size = 0
    if not os.path.exists(path):
        return 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except (OSError, IOError):
                pass
    return total_size

def format_size(size_bytes):
    """
    Formata um tamanho em bytes para uma string legível.
    
    Args:
        size_bytes: Tamanho em bytes
    
    Returns:
        String formatada (ex: "1.5 GB")
    """
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {size_names[i]}"

def delete_model(model_name, cache_dir):
    """
    Exclui um modelo específico do cache do HuggingFace.
    
    Args:
        model_name: Nome do modelo a ser excluído
        cache_dir: Diretório base do cache do HuggingFace
    
    Returns:
        True se o modelo foi excluído com sucesso, False caso contrário
    """
    model_path = os.path.join(cache_dir, 'hub', model_name)
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo '{model_name}' não encontrado!")
        return False
    
    try:
        size_before = get_directory_size(model_path)
        shutil.rmtree(model_path)
        print(f"✅ Modelo '{model_name}' excluído com sucesso!")
        print(f"   Espaço liberado: {format_size(size_before)}")
        return True
    except Exception as e:
        print(f"❌ Erro ao excluir modelo '{model_name}': {e}")
        return False

def list_models_by_size(cache_dir):
    """
    Lista todos os modelos no cache ordenados por tamanho (maior primeiro).
    
    Args:
        cache_dir: Diretório base do cache do HuggingFace
    
    Returns:
        Lista de tuplas (nome_do_modelo, tamanho_em_bytes)
    """
    hub_dir = os.path.join(cache_dir, 'hub')
    if not os.path.exists(hub_dir):
        print("❌ Diretório hub não encontrado!")
        return []
    
    models = []
    for item in os.listdir(hub_dir):
        item_path = os.path.join(hub_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            size = get_directory_size(item_path)
            models.append((item, size))
    
    # Ordenar por tamanho (maior primeiro)
    models.sort(key=lambda x: x[1], reverse=True)
    return models

def main():
    """
    Função principal do script de gerenciamento de modelos.
    
    Exibe uma interface interativa para listar e excluir modelos do cache
    do HuggingFace, permitindo escolher modelos específicos ou excluir
    todos os modelos grandes ou todos os modelos.
    """
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    
    print("=== GERENCIADOR DE MODELOS HUGGING FACE ===\n")
    
    # Listar modelos disponíveis
    models = list_models_by_size(cache_dir)
    
    if not models:
        print("❌ Nenhum modelo encontrado!")
        return
    
    print("Modelos instalados (ordenados por tamanho):")
    print("-" * 60)
    for i, (model_name, size) in enumerate(models, 1):
        print(f"{i:2d}. {format_size(size):>10} - {model_name}")
    
    print(f"\nTotal de modelos: {len(models)}")
    total_size = sum(size for _, size in models)
    print(f"Espaço total ocupado: {format_size(total_size)}")
    
    print("\n" + "="*60)
    print("OPÇÕES DE EXCLUSÃO:")
    print("1. Excluir modelo específico (digite o número)")
    print("2. Excluir todos os modelos grandes (>5GB)")
    print("3. Excluir todos os modelos")
    print("4. Sair")
    
    while True:
        try:
            choice = input("\nEscolha uma opção (1-4): ").strip()
            
            if choice == '4':
                print("Saindo...")
                break
            elif choice == '1':
                try:
                    model_num = int(input("Digite o número do modelo para excluir: "))
                    if 1 <= model_num <= len(models):
                        model_name = models[model_num - 1][0]
                        confirm = input(f"Tem certeza que deseja excluir '{model_name}'? (s/N): ")
                        if confirm.lower() in ['s', 'sim', 'y', 'yes']:
                            delete_model(model_name, cache_dir)
                        else:
                            print("Operação cancelada.")
                    else:
                        print("❌ Número inválido!")
                except ValueError:
                    print("❌ Digite um número válido!")
            elif choice == '2':
                large_models = [m for m in models if m[1] > 5 * 1024**3]  # >5GB
                if not large_models:
                    print("❌ Nenhum modelo grande encontrado!")
                    continue
                
                print(f"\nModelos grandes encontrados ({len(large_models)}):")
                for model_name, size in large_models:
                    print(f"  {format_size(size):>10} - {model_name}")
                
                confirm = input(f"\nExcluir todos os {len(large_models)} modelos grandes? (s/N): ")
                if confirm.lower() in ['s', 'sim', 'y', 'yes']:
                    deleted_count = 0
                    for model_name, _ in large_models:
                        if delete_model(model_name, cache_dir):
                            deleted_count += 1
                    print(f"\n✅ {deleted_count} modelos excluídos com sucesso!")
                else:
                    print("Operação cancelada.")
            elif choice == '3':
                confirm = input(f"⚠️  ATENÇÃO: Excluir TODOS os {len(models)} modelos? (s/N): ")
                if confirm.lower() in ['s', 'sim', 'y', 'yes']:
                    deleted_count = 0
                    for model_name, _ in models:
                        if delete_model(model_name, cache_dir):
                            deleted_count += 1
                    print(f"\n✅ {deleted_count} modelos excluídos com sucesso!")
                else:
                    print("Operação cancelada.")
            else:
                print("❌ Opção inválida! Digite 1, 2, 3 ou 4.")
                
        except KeyboardInterrupt:
            print("\n\nOperação cancelada pelo usuário.")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
