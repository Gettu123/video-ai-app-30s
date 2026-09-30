# Full-Stack Mobile & AI Video App (PocketBase + n8n + Image-to-Video + APK Builder)

Sistema completo desacoplado, autoalojado (Self-Hosted) y sin cuotas para generación de videos de hasta 30 segundos a partir de imágenes, grabación en vivo, autenticación segura JWT y compilación automática de APK en GitHub Actions.

## 📁 Estructura del Repositorio

- **`.github/workflows/build-apk.yml`**: Compilación CI/CD automática en GitHub Actions para generar el archivo instalador `.apk` de Android con Capacitor.
- **`docker/`**:
  - `docker-compose.yml`: Orquestación de PocketBase (Auth/DB), n8n (Motor de flujos), AI Video Worker (API de inferencia Image-to-Video) y Caddy (Reverse Proxy con SSL).
  - `Caddyfile`: Configuración de dominios y SSL automático.
- **`ai-worker/`**: Microservicio Python (FastAPI) preparado para conectarse a modelos Open Source de GitHub (Deforum / Stable Video Diffusion / LivePortrait / CogVideoX / ComfyUI) para generar videos de hasta 30 segundos a partir de una imagen.
- **`frontend/`**: Aplicación web móvil lista para compilar a APK (HTML5 + Tailwind + MediaRecorder + Image Upload + PocketBase Auth + n8n Handshake).
- **`n8n-workflows/`**: Flujo JSON de n8n con nodo Webhook, validación de JWT contra PocketBase, llamada al worker de IA y respuesta estructurada.

## 🚀 Despliegue Rápido en VPS

1. Copiar la carpeta `docker/` a tu servidor VPS.
2. Configurar tus dominios en `docker/Caddyfile` y las variables de entorno en `docker-compose.yml`.
3. Ejecutar:
   ```bash
   docker compose up -d
   ```
4. Acceder al panel de PocketBase (`https://pb.tu-dominio.com/_/`) y n8n (`https://n8n.tu-dominio.com`).

## 📱 Compilación del APK para Android

1. Sube este repositorio a GitHub.
2. Ve a la pestaña **Actions** en GitHub.
3. El workflow `Build Android APK` compilará el código y generará el artefacto `app-debug.apk` listo para descargar e instalar en el celular.

## Checklist del primer deploy

1. `cp docker/.env.example docker/.env` y completa `PB_ENCRYPTION_KEY` (exactamente 32 caracteres; `openssl rand -hex 16`), `PB_ADMIN_EMAIL` y `PB_ADMIN_PASSWORD`.
2. Ajusta `PB_DOMAIN`, `N8N_DOMAIN` y `VIDEO_DOMAIN` en ese `.env`. El `Caddyfile` los lee como variables.
3. Apunta el DNS de los tres hostnames a la VPS.
4. Desde `docker/`: `docker compose up -d`.
5. Entra al panel de PocketBase y crea un usuario en la coleccion `users` (el login de la app usa email y contrasena). El superusuario de `.env` solo abre el panel `/_/`.
6. En n8n, importa `n8n-workflows/video-ai-workflow.json` y **activa** el workflow. El webhook queda en `/webhook/ai-video`.
7. En la app, configura la URL de PocketBase y la del webhook.

El worker escribe un MP4 de hasta 30 segundos y lo publica en `https://VIDEO_DOMAIN/outputs/<job>.mp4`. Una imagen se anima con Ken Burns (FFmpeg) y el prompt se guarda al lado del job para conectar despues Deforum, SVD, LivePortrait, CogVideoX o ComfyUI. Un video de la camara se recorta a la duracion pedida.

## Notas del APK

El workflow de Actions usa Node 22, JDK 21, Android SDK 36 y Capacitor 8.5.2. El `webDir` es `frontend/www` (Capacitor 8 no acepta `.`). La carpeta `frontend/android/` se genera en CI y no se versiona. El artefacto descargable se llama `VideoAIApp-Debug-APK`.
