#!/usr/bin/env python3

from typing import Tuple, TypeVar

T = TypeVar('T')

# NOTE(ycho):
# Beware when using TupleN... alias together with
# simple_parsing, which causes issues with get_args().

Tuple1 = Tuple[T]
Tuple2 = Tuple[T, T]
Tuple3 = Tuple[T, T, T]
Tuple4 = Tuple[T, T, T, T]
Tuple5 = Tuple[T, T, T, T, T]
Tuple6 = Tuple[T, T, T, T, T, T]
Tuple7 = Tuple[T, T, T, T, T, T, T]
Tuple8 = Tuple[T, T, T, T, T, T, T, T]
Tuple9 = Tuple[T, T, T, T, T, T, T, T, T]

TranslationT = TypeVar('Translation', bound=Tuple3[float])
QuaternionT = TypeVar('Quaternion', bound=Tuple4[float])
PoseT = TypeVar('Pose', bound=Tuple[TranslationT, QuaternionT])
