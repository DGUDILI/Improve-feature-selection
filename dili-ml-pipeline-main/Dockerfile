FROM continuumio/miniconda3

WORKDIR /app

COPY docker/environment.yml .

# conda 환경 먼저 생성
RUN conda env create -f environment.yml

# pip는 따로 설치 (디버깅 가능하게)
SHELL ["conda", "run", "-n", "dili_ml_pipeline_env", "/bin/bash", "-c"]


COPY . .

CMD ["bash"]