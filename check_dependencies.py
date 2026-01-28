#!/usr/bin/env python3
"""
Dependency Checker Script
Verifica quais dependências do requirements.txt estão instaladas
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Verifica todas as dependências listadas em requirements.txt"""
    
    requirements_files = [
        'requirements.txt',
        'rag_requirements.txt',
    ]
    
    all_packages = {}
    
    # Ler todos os requirements files
    for req_file in requirements_files:
        if Path(req_file).exists():
            print(f"\n📋 Lendo {req_file}...")
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        all_packages[line] = req_file
    
    if not all_packages:
        print("❌ Nenhum requirements.txt encontrado!")
        return False
    
    print(f"\n📦 Total de pacotes a verificar: {len(all_packages)}\n")
    
    installed = []
    missing = []
    
    # Verificar cada pacote
    for package, source in all_packages.items():
        # Extrair nome do pacote (antes de ==, >=, etc)
        package_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].split('[')[0].strip()
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', package_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Extrair versão instalada
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':', 1)[1].strip()
                        installed.append((package, version, source))
                        print(f"✅ {package:<40} v{version:<10} ({source})")
                        break
            else:
                missing.append((package, source))
                print(f"❌ {package:<40} NÃO INSTALADO ({source})")
                
        except subprocess.TimeoutExpired:
            missing.append((package, source))
            print(f"⏱️  {package:<40} TIMEOUT ({source})")
        except Exception as e:
            missing.append((package, source))
            print(f"⚠️  {package:<40} ERRO: {str(e)}")
    
    # Relatório final
    print(f"\n{'='*70}")
    print(f"📊 RELATÓRIO FINAL")
    print(f"{'='*70}")
    print(f"✅ Instalados: {len(installed)}")
    print(f"❌ Faltando:  {len(missing)}")
    print(f"📦 Total:     {len(all_packages)}")
    
    if missing:
        print(f"\n⚠️  Pacotes que precisam ser instalados:")
        for package, source in missing:
            print(f"   • {package} (de {source})")
        
        print(f"\n💡 Para instalar os pacotes faltantes, execute:")
        missing_names = ' '.join([p[0].split('==')[0].split('>=')[0] for p in missing])
        print(f"   pip install {missing_names}")
        
        return False
    else:
        print(f"\n✨ Todas as dependências estão instaladas!")
        return True


if __name__ == '__main__':
    success = check_dependencies()
    sys.exit(0 if success else 1)
    sys.exit(0 if success else 1)
