import requests
from requests.auth import HTTPBasicAuth
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random
import difflib

def send_mail(buff):
    fromaddr="quiquemelle.xavier@gmail.com"
    toaddr="xavier.quiquemelle@gmail.com"
    body=buff
    msg=MIMEMultipart()
    msg['From']=fromaddr
    msg['To']=toaddr
    msg['Subject']="monitor DBUFR"
    msg.attach(MIMEText(body,'plain'))
    server=smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(fromaddr,"vieuxbilly92")
    text=msg.as_string(unixfrom=True)
    server.send_message(msg)
    server.quit()

url="https://www-dbufr.ufr-info-p6.jussieu.fr/lmd/2004/master/auths/seeStudentMarks.php"
auth=HTTPBasicAuth("3520742","Vieuxbilly92@")


b=True
while b:
    try:
        r=requests.get(url=url,auth=auth)
        b=False
    except Exception as e:
        print(e)
        print("bug requests")
        time.sleep(10)
buff=r.text.split("Et voici vos notes publiées")[1]
time.sleep(120)
bip=0



while True:
    b=True
    while b:
        try:
            rr=requests.get(url=url,auth=auth)
            b=False
        except Exception as e:
            print(e)
            time.sleep(10)
    if rr.text.split("Et voici vos notes publiées")[1]!=buff:


        case_a = rr.text
        case_b = buff

        output_list = [li for li in difflib.ndiff(case_a, case_b) if li[0] != ' ']
        send_mail("Nouvelle note {}\n\n{}".format(url,output_list))

        buff=rr.text.split("Et voici vos notes publiées")[1]
        print("bravo")
    bip+=1
    print("bip {}".format(bip))
    time.sleep(random.randint(120,240))
    
