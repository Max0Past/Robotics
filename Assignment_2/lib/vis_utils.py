import matplotlib.pyplot as plt


def visualize_images(imgs, texts):
    NUM_IMGS = 18
    ROWS = 3
    COLS = 6
    assert ROWS * COLS == NUM_IMGS
    fig = plt.figure()
    for i in range(NUM_IMGS):
        fig.add_subplot(ROWS, COLS, i+1)
        img = imgs[i]
        if img.shape[-1] > 3:
            img = img.transpose(1, 2, 0)
        plt.imshow(img) 
        plt.axis('off') 
        plt.title(texts[i]) 


def visualize_predictions(imgs, preds, labels, class_names):
    texts = []
    for i in range(preds.shape[0]):
        pred_text = class_names[preds[i]]
        label_text = class_names[labels[i]]
        texts.append(f"{pred_text}\n[{label_text}]")
    visualize_images(imgs, texts)


def visualize_dataset_images(imgs, labels, class_names):
    texts = []
    for i in range(labels.shape[0]):
        label_text = class_names[labels[i]]
        texts.append(f"{label_text}")
    visualize_images(imgs, texts)