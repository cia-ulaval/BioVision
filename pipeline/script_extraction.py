import  cv2
from pathlib import Path



## Les images et vidéos seront stockées dans le dossier data 
## 1. Les vidéos sont dans un sous-dossier 'Videos'. 
## 2. Pour chacune des vidéos:
## - Créer un sous-dossier dans 'Images'
## - Lire et capturer la vidéo
## - Extraire les frames 
## - Enregistrer chaque frame extraite dans le sous-dossier créé
## 3. Répéter pour chacune des vidéos dans 'Videos'


 ## extraire les images de la vidéo
def extraireimages(video_path: Path, sortie_dir: Path):     

    ## créer un sous-dossier dans Images
    sortie_dir.mkdir(parents=True, exist_ok=True)      

    ## comme pour "lire" une vidéo
    capture = cv2.VideoCapture(str(video_path))    
    fps = capture.get(cv2.CAP_PROP_FPS)
    pas = max(1, round(fps))

   ## initialiser le compteur, début de la vidéo
    framecourante = 0               

    ## tant que la vidéo est ouverte
    while capture.isOpened():   

        ## capturer frame par frame
        ret,frame = capture.read()         


        ## si la vidéo n'est pas finie
        if ret:                          
            ## Donner un nom à la frame
            name = sortie_dir / (str(framecourante)+'.jpg')   
                       
            if framecourante % pas == 0:
                cv2.imwrite(name,frame)
                
            framecourante = framecourante + 1

        else:
            break

    capture.release()   
    
    
    

def extraire(dossier_video, dossier_sortie):

    ## utiliser une boucle for pour capturer et extraire des images de chacune des vidéos dans le sous-dossier 
    
    for video in Path(dossier_video).glob("*"):  ## Parcourir toutes les vidéos
        extraireimages(video, Path(dossier_sortie)/video.stem)   ## appeler la fonction d'extraction

