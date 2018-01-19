from urllib.request import urlopen
html = urlopen("http://google.eu")
print(html.read())
fetch ()