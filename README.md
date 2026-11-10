## Installation

L'environnement de développement de BioVision tourne entièrement dans Docker. Toute l'équipe utilise ainsi les mêmes versions de Python et des bibliothèques (PyTorch, OpenCV, Ultralytics, etc.). Inutile d'installer Python ou un environnement virtuel sur votre machine.

### Prérequis

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS)

> Sur Windows, l'installation de Docker Desktop peut demander un redémarrage. Au premier lancement, acceptez les conditions ; la connexion à un compte Docker est facultative (*Skip*).

### 1. Démarrer Docker Desktop

Les commandes `docker` ne fonctionnent que si Docker Desktop est lancé. Vous n'avez pas besoin d'utiliser son interface : il doit seulement tourner en arrière-plan.

- **Windows** : ouvrez le menu Démarrer, cherchez **Docker Desktop** et lancez-le.
- **macOS** : ouvrez **Docker** depuis le dossier Applications (ou avec Spotlight : `Cmd + Espace`, puis « Docker »).

Attendez que l'icône de la baleine 🐳 dans la barre des tâches (Windows) ou la barre de menus (macOS) arrête de s'animer. Vérifiez ensuite dans un terminal que Docker répond :

```bash
docker info
```

Si la commande affiche `Cannot connect to the Docker daemon`, Docker Desktop n'est pas encore démarré : attendez quelques secondes et réessayez.

> Astuce : pour ne plus avoir à le lancer à la main, activez *Settings → General → Start Docker Desktop when you sign in*.

### 2. Cloner le dépôt

```bash
git clone https://github.com/cia-ulaval/BioVision.git
cd BioVision
```

### 3. Construire et lancer le conteneur

```bash
docker compose up --build
```

Laissez ce terminal ouvert : il affiche les logs et le conteneur s'arrête si vous le fermez.

La première construction prend plusieurs minutes. Ensuite, `docker compose up` suffit.

> Reconstruisez l'image (`docker compose up --build`) seulement si `requirements.txt` ou le `Dockerfile` change.

Une fois le conteneur démarré, ouvrez JupyterLab sur **http://localhost:8888**

### 4. Arrêter l'environnement

Faites `Ctrl + C` dans le terminal du conteneur, ou lancez la commande suivante dans un autre terminal, depuis le dossier `BioVision` :

```bash
docker compose down
```
