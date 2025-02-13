from celery import Celery
import subprocess

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def DTL_predict(weight,data):
    # 调用预测脚本使用相应权重和数据进行预测
    process_cmd = ['python', './model/DeepTraLog-code/DeepTraLog.py', 'predict','--weight','--data']
    subprocess.run(process_cmd, check=True)