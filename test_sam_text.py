from ultralytics import SAM
model = SAM("/Users/rickyho/Documents/github/sam/sam3.pt")
results = model.predict("/Users/rickyho/Documents/github/sam/data/IMG_20260901_141556.jpg", texts=["tree crown"])
print(results[0].masks)
