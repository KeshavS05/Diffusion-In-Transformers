from my_models import DiT as MyDiT

def _build(input_size: int, num_classes: int):
    dim, num_blocks, heads = (384, 12, 6)
    y_dim = max(num_classes, 1)
    model = MyDiT(p=2, d=dim, blocks=num_blocks, t_dim=256, y_dim=y_dim, in_ch=4, type="")
    return model

DiT_models = {
    "DiT-S/2":  lambda input_size, num_classes: _build(input_size, num_classes),
}
