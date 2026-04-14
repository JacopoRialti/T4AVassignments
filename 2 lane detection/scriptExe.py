import os
import subprocess
import glob
import sys

def main():
    # Dai log del terminale, l'archivio si trova sotto "archive/archive/" nel progetto 
    archive_dir = os.path.join("archive", "archive")
    if not os.path.exists(archive_dir):
        # Fallback se è solo "archive/"
        archive_dir = os.path.join("archive")
    
    # Trova tutte le sottocartelle numerate
    folders = sorted(glob.glob(os.path.join(archive_dir, "*/")))
    print(f"Trovate {len(folders)} cartelle in '{archive_dir}'.")

    if not folders:
        print(f"Nessuna cartella trovata. Verifica che i percorsi del dataset siano corretti.")
        return

    for folder in folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        camera_dir = os.path.join(folder, "camera", "front_camera")
        
        # IMPORTANTE: run_gold.py accetta un GLOB pattern, non solo la cartella!
        glob_path = os.path.join(camera_dir, "*.jpg")
        
        if os.path.isdir(camera_dir):
            print(f"\n{'='*50}")
            print(f"Avvio elaborazione per la sequenza: {folder_name}")
            print(f"{'='*50}")
            
            # Lancia il tuo script run_gold.py passandogli il pattern glob/*.jpg
            # sys.executable garantisce che venga usato il 'python' dell'ambiente virtuale attivo (venv)
            # cwd=project_dir fa in modo che il terminale sia posizionato temporaneamente dove risiede run_gold.py
            subprocess.run([sys.executable, "run_gold.py", glob_path])
            
            # Quando run_gold.py termina (dopo che chiudi la sua finestra col tasto ESC o 'q'),
            # lo script si mette in pausa e aspetta un tuo comando
            try:
                input("\n👉 Premi INVIO per passare al prossimo video, o premi Ctrl+C per interrompere la coda... ")
            except KeyboardInterrupt:
                print("\nEsecuzione interrotta dall'utente.")
                break
        else:
            print(f"⚠️ Cartella 'camera/front_camera' non trovata in {folder_name}, salto la sequenza.")

if __name__ == "__main__":
    main()
