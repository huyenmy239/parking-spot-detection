import cv2
import pickle

width, height = 210, 158

try:
    with open("carPos-2", "rb") as f:
        posList = pickle.load(f)
except:
    posList = []

def getSpotLabel(index):
    row = index // 10
    col = index % 10
    return f"{chr(65 + row)}{col + 1}"

def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)

    with open("carPos-2", "wb") as f:
        pickle.dump(posList, f)

cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    img = cv2.imread("car-spot.png")

    for i, pos in enumerate(posList):
        x, y = pos
        label = getSpotLabel(i)

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, label, (x + 5, y + height - 5), font, 0.8, (255, 0, 255), 2)

    cv2.imshow("image", img)
    cv2.setMouseCallback("image", mouseClick)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
