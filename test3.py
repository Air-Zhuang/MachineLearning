import pymysql

contact=0
sales=0
music=0
info=0
hello=0
advice=0
db = pymysql.connect()
cur = db.cursor()
try:
    cur.execute("select count(*) from netizens;")
    result = cur.fetchone()
    num=int(result[0])
    cur.execute("select email from netizens;")
    results = cur.fetchall()
    for i in results:
        if i[0].startswith("contact@"):
            contact+=1
        if i[0].startswith("sales@"):
            sales+=1
        if i[0].startswith("music@"):
            music+=1
        if i[0].startswith("info@"):
            info+=1
        if i[0].startswith("hello@"):
            hello+=1
        if i[0].startswith("advice@"):
            advice+=1
except Exception as e:
    raise e
finally:
    db.close()

print("总数量：",num)
print("contact:",contact,"占比：",str(round((contact/num)*100,2))+"%")
print("sales:",sales,"占比：",str(round((sales/num)*100,2))+"%")
print("music:",music,"占比：",str(round((music/num)*100,2))+"%")
print("info:",info,"占比：",str(round((info/num)*100,2))+"%")
print("hello:",hello,"占比：",str(round((hello/num)*100,2))+"%")
print("advice:",advice,"占比：",str(round((advice/num)*100,2))+"%")