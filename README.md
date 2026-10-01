# Full-Stack Mobile & AI Video App (PocketBase + n8n + Image-to-Video + APK Builder)

Sistema completo desacoplado, autoalojado (Self-Hosted) y sin cuotas para generación de videos de hasta 30 segundos a partir de imágenes, grabación en vivo, autenticación segura JWT y compilación automática de APK en GitHub Actions.

## 📁 Estructura del Repositorio

- **`.github/workflows/build-apk.yml`**: Compilación CI/CD automática en GitHub Actions para generar el archivo instalador `.apk` de Android con Capacitor.
- **`docker/`**:
  - `docker-compose.yml`: Orquestación de PocketBase (Auth/DB), n8n (Motor de flujos), AI Video Worker (API de inferencia Image-to-Video) y Caddy (Reverse Proxy con SSL).
  - `Caddyfile`: Configuración de dominios y SSL automático.
- **`ai-worker/`**: FastAPI. Ken Burns (sin GPU) o, con el perfil NVIDIA, Wan2.1 I2V, HunyuanVideo, HunyuanVideo-I2V y CogVideoX / CogVideoX1.5. VideoX-Fun encadena el ultimo cuadro hasta 30s.
- **`docker/docker-compose.gpu.yml`**: mismo stack con el worker en PyTorch CUDA.
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

El worker escribe un MP4 de hasta 30 segundos y lo publica en `https://VIDEO_DOMAIN/outputs/<job>.mp4`.

## Motores

Ninguno de estos modelos suelta 30s en un solo forward. El modo **VideoX-Fun** repite la idea de [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun): el ultimo cuadro de un clipe es la imagen del siguiente, y FFmpeg concatena.

| Motor | Repo | Clipe nativo | Notas |
| --- | --- | --- | --- |
| Ken Burns | FFmpeg local | 30s | La imagen slim. No usa GPU. |
| Wan2.1 I2V 14B | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | 81 frames @ 16fps (~5s) | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`. Apache-2.0. |
| HunyuanVideo-I2V | [Tencent-Hunyuan/HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V) | 129 frames @ 24fps (~5s) | Pesos Diffusers `hunyuanvideo-community/HunyuanVideo-I2V`. |
| HunyuanVideo | [Tencent-Hunyuan/HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) | ~5s | Texto a video. La extension usa I2V. |
| CogVideoX-5B I2V | [zai-org/CogVideo](https://github.com/zai-org/CogVideo) | 49 frames @ 8fps (~6s) | `THUDM/CogVideoX-5b-I2V`. |
| CogVideoX1.5 I2V | [zai-org/CogVideo](https://github.com/zai-org/CogVideo) | 81 frames @ 8fps (~10s) | `THUDM/CogVideoX1.5-5B-I2V`. Tres clipes cubren 30s. |

Los pesos no van en la imagen Docker. La primera peticion los baja al volumen `hf_cache`. Hunyuan y CogVideoX piden aceptar la licencia en Hugging Face y, si el repo es gated, `HF_TOKEN` en `docker/.env`.

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

`GET /engines` en el dominio de video dice que motor tiene CUDA. Despues de cambiar el workflow, vuelve a importar `n8n-workflows/video-ai-workflow.json` (el nodo ahora reenvia `engine`, `base_engine` y `chain`, con timeout de 60 minutos).

## Notas del APK

El workflow de Actions usa Node 22, JDK 21, Android SDK 36 y Capacitor 8.5.2. El `webDir` es `frontend/www` (Capacitor 8 no acepta `.`). La carpeta `frontend/android/` se genera en CI y no se versiona. El artefacto descargable se llama `VideoAIApp-Debug-APK`. En un Galaxy Ultra se instala como APK de depuración (orígenes desconocidos). En el teléfono funcionan la cámara, el Ken Burns de 30s a 720p y guardar el video en los 512 GB. Wan2.1, HunyuanVideo y CogVideoX no corren en el aparato: siguen en el servidor con perfil GPU.
