from enum import StrEnum


class Opcode(StrEnum):
    TENSOR = "tensor"
    ADD = "add"
    MUL = "mul"
    COPY = "copy"
    CONCAT = "concat"
    MATMUL = "matmul"
    SLICE = "slice"
    FUSE = "fuse"
