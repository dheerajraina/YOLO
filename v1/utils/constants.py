architecture = [
    (7, 64, 2),  # (kernel_size,filters,stride)
    "Pooling",  # Represents a max-pooling layer => (pool_size=(2,2),stride=2)
    (3, 192, 1),
    "Pooling",
    (1, 128, 1),
    (3, 256, 1),
    (1, 256, 1),
    (3, 512, 1),
    "Pooling",
    # last element represents the number of times the given conv blocks need to be repeated
    [(1, 256, 1), (3, 512, 1), 4],
    (1, 512, 1),
    (3, 1024, 1),
    "Pooling",
    [(1, 512, 1), (3, 1024, 1), 2],
    (3, 1024, 1),
    (3, 1024, 2),
    (3, 1024, 1),
    (3, 1024, 1),
]