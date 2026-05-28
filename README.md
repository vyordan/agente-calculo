[![Docker Image Size](https://img.shields.io/docker/image-size/vyordan/agente-calculo?label=Image%20Size)](https://hub.docker.com/r/vyordan/agente-calculo)
[![Docker Pulls](https://img.shields.io/docker/pulls/vyordan/agente-calculo?label=Pulls)](https://hub.docker.com/r/vyordan/agente-calculo)


## Ejecutar
Antes de ejecutar los siguientes ocmandos es muy importante tener un entorno virtual y activarlo, por el tema de las dependencias :)
```bash
# Instalar dependencias
pip install -r requirements.txt

# Descargar los modelos
python download_models.py

# Iniciar la applicacion, una vez se tienen las dependencias y los
#modelos descargados solo es necesario este comando para volver a ejecutar
streamlit run src/app.py 
```

