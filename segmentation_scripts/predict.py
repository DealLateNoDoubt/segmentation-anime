from ultralytics import YOLO

if __name__ == "__main__":

    # Load a model
    # model = YOLO("./runs/m-yolo-seg/weights/best.pt")
    # model = YOLO("./runs/m-start-ep100/weights/best.pt")
    # model = YOLO("./runs/m-starnet-glsa-bifpn-ep100/weights/best.pt")
    model = YOLO("./runs/m-starnet-glsa-bifpn-ep100-1/weights/best.pt")
    model.predict(
        source=r'F:\图集\17702964_PPPP\115174952_🤍💥_.jpg',
        imgsz=640,
        save=False,
        show=True,
    )