import re
res = re.search('https*:[//\w.]+',".... https://www.google.com")
print(res.group()) #https://www.google.com