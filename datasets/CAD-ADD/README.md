# CAD-ADD Mirror

This folder mirrors the dataset layout expected by the TDRE training scripts.

Expected structure:
- `Agricultural Detection/<weather>/{train,test}`
- `Agricultural Detection/Labels/{train,test}`
- `Rescue Detection/<weather>/{train,test}`
- `Rescue Detection/Labels/{train,test}`
- `Waste Detection/<weather>/{train,test}`
- `Waste Detection/Labels/{train,test}`
- `Transport Detection/<weather>/{train,test}`
- `Transport Detection/Labels/{train,test}`
- `Real Transport Detection/images`
- `Real Transport Detection/Labels`

Copy or symlink the actual CAD-ADD files here before training if you want to use the default local path.
