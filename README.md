# 🎯 Détection d'Objets en Temps Réel avec YOLOv8, OBS et Flask

## 📌 Description

Ce projet met en œuvre une application de **détection d'objets en temps réel** basée sur le modèle **YOLOv8** (You Only Look Once). Il utilise **OBS Studio** pour capturer le flux vidéo et **Flask** pour fournir une interface web en direct affichant les résultats de détection.

L'objectif est de démontrer comment combiner un modèle de vision par ordinateur avancé avec un serveur web léger pour créer une application de surveillance ou d'analyse intelligente.

---

## 🛠️ Technologies Utilisées

- 🔍 **YOLOv8** — Détection d'objets rapide et précise (via la bibliothèque Ultralytics)
- 📹 **OBS Studio** — Capture de flux vidéo (webcam ou écran)
- 🌐 **Flask** — Interface web pour visualiser la détection en temps réel
- 🐍 **Python 3.8+**
- 🧠 **OpenCV** — Traitement d'image
- 📦 **ultralytics**, **flask**, **opencv-python**, etc.

---

## 🖼️ Fonctionnement

1. **OBS** capture un flux vidéo depuis la caméra ou une source écran.
2. Le flux est traité par un script Python utilisant **YOLOv8** pour détecter les objets.
3. Une interface **Flask** affiche le flux vidéo avec les objets détectés en temps réel.

---

## 📁 Structure du Projet

