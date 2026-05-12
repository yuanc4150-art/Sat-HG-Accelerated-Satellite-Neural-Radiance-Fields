
from .satnerf_hashgrid import HashGridNeRF


def load_model(args, aabb=None):

    model_name = args.model.lower()

    if model_name == "sat-nerf-ngp":

        if aabb is None:
            raise ValueError("AABB (Axis-Aligned Bounding Box) must be provided to initialize the HashGridNeRF model.")


        model = HashGridNeRF(
            aabb_min=aabb[0],
            aabb_max=aabb[1],
            number_of_outputs=9,
            mlp_hidden=64,
            mlp_layers=2,
            occ_resolution=128,
            occ_tau=0.01,



            args=args

        )

    else:
        raise ValueError(f"Model '{args.model}' is not a valid model name.")

    return model