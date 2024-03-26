import enum
from copy import copy
from pathlib import Path
from typing import Optional, Set, Dict, Any, Type, List, Union, Callable

import numpy as np

from utilities.enumerations import DownSampleMethod as DSM
from utilities.enumerations import MeshCleaningMethod as MCM
from utilities.enumerations import MeshEvaluationMetric as MEM
from utilities.enumerations import SurfaceReconstructionMethod as SRM
from utilities.enumerations import SurfaceReconstructionParameters as SRP
from utilities.enumerations import TriangleNormalDeviationMethod as TNDM


class RunConfiguration:
    def __init__(self):
        self.name: str = "Unnamed run configuration"
        # Point Cloud Settings (incl. down sampling)
        self.point_cloud_path: Path = None
        self.down_sample_method: Optional[DSM] = None
        self.down_sample_params: float = None

        # Point cloud Normal Estimation Settings
        self.normal_estimation_neighbours: int = 0
        self.normal_estimation_radius: float = 0.0
        self.skip_normalizing_normals: bool = False
        self.orient_normals: Optional[int] = None

        # Surface Reconstruction Settings
        self.surface_reconstruction_method: SRM = SRM.DELAUNAY_TRIANGULATION
        self.alpha: float = 0.0
        self.ball_pivoting_radii: Union[list, set, tuple] = []
        self.poisson_density_quantile = 0.1
        self.poisson_octree_max_depth = 8

        # Mesh Cleaning Settings
        self.mesh_cleaning_methods: Optional[Set[MCM]] = None
        self.edge_length_cleaning_portion: float = 1.0
        self.aspect_ratio_cleaning_portion: float = 1.0

        # Mesh Evaluation
        self.mesh_quality_metrics: Optional[Set[MEM]] = None
        self.triangle_normal_deviation_method: TNDM = TNDM.NAIVE

        # Misc settings
        self.store_mesh: bool = False
        self.store_preprocessed_pointcloud: bool = False
        self.preprocessed_pointcloud_path: Optional[Union[str, Path]] = None
        self.processes: int = 1
        self.chunk_size: int = 1000

        self.overwritten_attributes: Set[str] = set()

    @property
    def surface_reconstruction_params(self) -> Dict[SRP, Any]:
        return {SRP.ALPHA: self.alpha,
                SRP.BPA_RADII: self.ball_pivoting_radii,
                SRP.POISSON_DENSITY_QUANTILE_THRESHOLD: self.poisson_density_quantile,
                SRP.POISSON_OCTREE_MAX_DEPTH: self.poisson_octree_max_depth}

    def __copy__(self):
        c = RunConfiguration()
        for var_name, value in self.__dict__.items():
            setattr(c, var_name, value)
        return c

    def copy(self):
        return self.__copy__()

    def set_setting(self,
                    data: dict,
                    setting_name: str,
                    run_config_name: Optional[str] = None,
                    default: Optional[Any] = None,
                    cast_method: Optional[Union[Type, Callable[[Any], Any]]] = None,
                    force_unique: bool = False,
                    handle_all_value: bool = False,
                    special_all_value: Optional[Any] = None,
                    special_all_values_getter: Optional[Union[Callable, enum.Enum, set, list, tuple]] = None,
                    verbose: bool = True,
                    ignore_default_if_overwritten: bool = True) -> None:
        """
        Set a setting in this run configuration.

        :param ignore_default_if_overwritten: If True and the setting name already exists in \
            `self.overwritten_attributes`, the `default` value will be ignored.
        :param special_all_values_getter: The value that determines the final value for the setting if the special \
            "all" value (specified by `special_all_value`) is found in or as the raw setting value. \
            If not specified (i.e. None), the final setting value will be the same as the special "all" value. \
            If a set, list, tuple or enum, the final setting value will be a copy of that. If a callable, the setting \
            value will be the result of calling the callable.
        :param handle_all_value: Whether to have a special "all" value that the found value collapses or is converted \
            to.
        :param special_all_value: The special "all" value to look for. If this special "all" value is found in the \
            raw setting value (when the value is an iterable) or if the raw setting value equals the special "all" \
            all value, the value of :parameter special_all_values_getter: will be used to determine the final value.
        :param force_unique: If true and the found value is an iterable, forces the iterable to a set. If false or \
            true and the found value is not an iterable, this is ignored.
        :param data: The dictionary containing the raw values.
        :param setting_name: The name of the setting to set.
        :param run_config_name: The name of the attribute of the run configuration. Set to None to use the setting name.
        :param default: The default value to use if the setting cannot be found in data.
        :param cast_method: Optional, either a callable to convert the value to the right value OR a type to cast \
            the value to. If the setting value is an accepted iterable (e.g. list), every element will be converted individually.
        :param verbose: Whether to print errors or warning messages.
        :return: None
        """

        if data is None:
            raise ValueError(f"Cannot set setting without provided data object.", self)

        if setting_name is None or len(setting_name) == 0:
            raise ValueError(f"Cannot set None or empty setting.", self)

        if run_config_name is None:
            run_config_name = setting_name

        if not hasattr(self, run_config_name) and verbose:
            raise ValueError(f"RunConfiguration has no attribute with name '{run_config_name}'", self)

        raw_value = None
        if setting_name in data:
            raw_value = data[setting_name]
            self.overwritten_attributes.add(setting_name)
        # If we are aiming for 'safe' execution, we do not write a default if it's already overwritten
        elif ignore_default_if_overwritten and setting_name in self.overwritten_attributes:
            return
        else:
            raw_value = default
            if verbose:
                print(f"Setting {setting_name} not found. Using default {default}.")

        if not isinstance(raw_value, (float, int, bool, list, enum.Enum, str)) and not (raw_value is None):
            raise ValueError(f"Value {raw_value} or default {default} for setting {setting_name} has to be of type "
                             f"{(float, int, bool, list, enum.Enum, str)}, but got {type(raw_value)}.", self)

        cast_method_active = not (cast_method is None) and isinstance(cast_method, (Callable, type))
        raw_value_is_iterable = isinstance(raw_value, (list, tuple, set, np.ndarray))

        # Casting
        # If the raw value is an accepted iterable, try to cast every value using the callable
        try:
            if cast_method_active and raw_value_is_iterable:
                raw_value = [cast_method(i) for i in raw_value]
            elif cast_method_active and not raw_value_is_iterable:
                raw_value = cast_method(raw_value)
        except Exception as e:
            raise e

        # If we need to handle a special "all" value, we try to 'detect' the special 'all' value.
        if handle_all_value:
            all_value_detected = raw_value_is_iterable and special_all_value in raw_value or special_all_value == raw_value
            if all_value_detected:  # If we detect this value in or as the raw value...
                if isinstance(special_all_values_getter, (list, set, tuple, np.ndarray)):
                    raw_value = copy(special_all_values_getter)  # Copy the values from the special all values getter
                elif isinstance(special_all_values_getter, (enum.Enum, enum.EnumMeta)):
                    raw_value = [i for i in special_all_values_getter]  # Convert the enum to a list.
                elif isinstance(special_all_values_getter, Callable):
                    raw_value = special_all_values_getter()  # Just call the special all values getter directly
                else:  # If no (valid) special all values getter is provided, just use the special all value itself.
                    raw_value = special_all_value

        raw_value_is_iterable = isinstance(raw_value, (list, tuple, set, np.ndarray))
        if raw_value_is_iterable and force_unique:
            raw_value = set(raw_value)

        try:
            setattr(self, run_config_name, raw_value)
        except Exception as e:
            raise e
