package = "word2vec"
version = "0.1-1"
source = {
    url = "https://github.com/prannayk/word2vec.git"
}
description = {
    summary = "Word to Vector representation neural network model",
    detailed = [[
        This is based on the code written for Torch and further written by Prannay Khosla and is available under the MIT LICENSE/X11.
    ]],
    license = "MIT/X11"
}
dependencies = {
    "lua >= 5.1, <5.4"
    "torch"
}
build = {
    type = "builtin"
    modules = {
        word2vec = "./init.lua"
    }
}
