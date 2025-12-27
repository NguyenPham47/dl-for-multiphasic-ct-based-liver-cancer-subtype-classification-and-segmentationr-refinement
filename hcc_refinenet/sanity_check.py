# After building train_loader in train.py (temporary)
b = next(iter(train_loader))
print("img:", b["img"].shape, "lab_sum:", b["lab"].sum().item(), "roi_sum:", b["roi"].sum().item())
