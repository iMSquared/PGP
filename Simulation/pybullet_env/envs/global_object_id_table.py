
from dataclasses import dataclass, field
import numpy.typing as npt
from typing import Dict, Tuple, List, Iterable, TypeVar, Callable
import numpy as np

Gid_T = TypeVar("Gid_T", bound=int)


class GlobalObjectIDTable:
    """A.k.a. gid table"""

    @dataclass(frozen=True)
    class ShapeInfoEntry:
        urdf_file_path: str         # Absolution
        pcd_file_path : str         # Absolute path
        pcd_points: npt.NDArray     # Index should match
        pcd_normals: npt.NDArray    # Index should match
        

    @dataclass(frozen=True)
    class Header:
        is_target: bool
        shape_info: "GlobalObjectIDTable.ShapeInfoEntry" 
        # NOTE(ssh): we will later work with list.
        

    def __init__(self):
        """TODO(ssh): This will later be taking the output from the initial belief generator..."""
        # Key of this dict represent the unique object. We just let 0, 1, 2, ... as key.
        self.__global_objects: Dict[Gid_T, "GlobalObjectIDTable.Header"] = {}
        self.__frozen = False


    def freeze(self):
        """Freeze dictionary"""
        self.__frozen = True

    
    def select_random_gid(self) -> Gid_T:
        """Select any random object"""
        # Select
        i = np.random.randint(0, len(self.__global_objects.keys()))
        selected = list(self.__global_objects.keys())[i]

        return selected


    def select_random_non_target_gid(self) -> Gid_T:
        """Select a random non-target object"""
        # Filter        
        func_filter: Callable[[Gid_T], bool] \
            = lambda gid: not self.__global_objects[gid].is_target
        non_target_gids = tuple(filter(func_filter, self.__global_objects.keys()))
        
        # Select
        i = np.random.randint(0, len(non_target_gids))
        selected = non_target_gids[i]

        return selected


    def select_target_gid(self) -> Gid_T:
        """Select a unique target object from the current object states"""

        # Filter        
        func_filter: Callable[[Gid_T], bool] \
            = lambda gid: self.__global_objects[gid].is_target
        target_gids = tuple(filter(func_filter, self.__global_objects.keys()))
        
        # Select
        if len(target_gids) != 1:
            raise ValueError("Multiple target objects in state")
        selected = target_gids[0]

        return selected



    # Just some wrapping functions to the dict
    def __setitem__(self, gid: Gid_T, header: Header):
        if self.__frozen:
            raise RuntimeError("Modifying the frozen table.")
        self.__global_objects[gid] = header

    def __getitem__(self, gid: Gid_T) -> Header:
        return self.__global_objects[gid]

    def keys(self) -> Iterable[Gid_T]:
        return self.__global_objects.keys()

    def values(self) -> Iterable[Header]:
        return self.__global_objects.values()

    def items(self) -> Iterable[Tuple[Gid_T, Header]]:
        return self.__global_objects.items()