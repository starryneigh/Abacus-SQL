conda activate demo


mkdir -p /root/text2sql-demo/model/Qwen2.5-Coder /root/text2sql-demo/model/SGPT

ln -s /root/hf_models/Qwen-0.5b /root/text2sql-demo/model/Qwen2.5-Coder/0.5b
ln -s /root/hf_models/SGPT-125M-weightedmean-msmarco-specb-bitfit /root/text2sql-demo/model/SGPT/125m
ln -s /root/quantized/Qwen2.5-Coder/7b /root/text2sql-demo/model/Qwen2.5-Coder/7b

streamlit run app.py

# while true; do
#     # 你的脚本逻辑
#     echo "Running..."
#     sleep 5  # 避免 CPU 占用过高
# done