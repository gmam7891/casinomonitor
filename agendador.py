from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from seu_script_principal import buscar_vods_twitch_por_periodo, TODOS_STREAMERS, HEADERS_TWITCH, BASE_URL_TWITCH, salvar_vods_no_banco

scheduler = BackgroundScheduler()

def tarefa_diaria():
    dt_ini = datetime.today() - timedelta(days=1)
    dt_fim = datetime.today()
    vods = buscar_vods_twitch_por_periodo(dt_ini, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, TODOS_STREAMERS)
    salvar_vods_no_banco(vods)

scheduler.add_job(tarefa_diaria, 'cron', hour=10, minute=0)
scheduler.start()

import time
try:
    while True:
        time.sleep(60)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
