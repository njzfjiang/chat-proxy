DATE=$(date +%Y%m%d_%H%M%S)
cd  ~/opt/chat-proxy
git add .
git commit -m "WB update on $DATE"
git push origin master
