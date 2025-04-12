FROM continuumio/miniconda3
WORKDIR /root/text2sql-demo
COPY ./text2sql-demo /root/text2sql-demo
COPY ./SGPT/125m /root/hf_models/SGPT-125M-weightedmean-msmarco-specb-bitfit
RUN conda create -y -n demo python=3.10
ENV PATH /opt/conda/envs/demo/bin:$PATH
RUN chmod +x ./slurm/setup.sh && /bin/bash ./slurm/setup.sh
EXPOSE 8501