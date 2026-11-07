# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def rmw_array_kernel(foos: wp.array[wp.uint32]):
    i = wp.tid()

    foos[i] += wp.uint32(1)


def test_rmw_array(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=rmw_array_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.struct
class RmwFoo:
    field: wp.uint32


@wp.kernel
def rmw_array_struct_kernel(foos: wp.array[RmwFoo]):
    i = wp.tid()
    foos[i].field += wp.uint32(1)


def test_rmw_array_struct(test, device):
    foos = wp.zeros((10,), dtype=RmwFoo, device=device)

    wp.launch(kernel=rmw_array_struct_kernel, dim=(10,), inputs=[foos], device=device)

    expected = RmwFoo()
    expected.field = 1
    for f in foos.list():
        test.assertEqual(f.field, expected.field)


@wp.func
def lookup(foos: wp.array[wp.uint32], index: int):
    return foos[index]


@wp.kernel
def lookup_kernel(foos: wp.array[wp.uint32]):
    i = wp.tid()

    x = lookup(foos, i)
    foos[i] = x + wp.uint32(1)


def test_lookup(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=lookup_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.func
def lookup3(foos: wp.array[wp.float32], index: int):
    return foos[index]


@wp.kernel
def grad_kernel(foos: wp.array[wp.float32], bars: wp.array[wp.float32]):
    i = wp.tid()

    x = lookup3(foos, i)
    bars[i] = x * wp.float32(i) + 1.0


def test_grad(test, device):
    num = 10
    data = np.linspace(20, 20 + num, num, endpoint=False, dtype=np.float32)
    input = wp.array(data, device=device, requires_grad=True)
    output = wp.zeros(num, dtype=wp.float32, device=device)

    ones = wp.array(np.ones(len(output)), dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=grad_kernel, dim=(num,), inputs=[input], outputs=[output], device=device)

    tape.backward(grads={output: ones})

    # test forward results
    for i, f in enumerate(output.list()):
        test.assertEqual(f, data[i] * i + 1)

    # test backward results
    for i, f in enumerate(tape.gradients[input].list()):
        test.assertEqual(f, i)


@wp.func
def lookup2(foos: wp.array[wp.uint32], index: int):
    if index % 2 == 0:
        x = foos[index]
        x = wp.uint32(0)
        return x
    else:
        return foos[index]


@wp.kernel
def lookup2_kernel(foos: wp.array[wp.uint32]):
    i = wp.tid()

    x = lookup2(foos, i)
    foos[i] = x + wp.uint32(1)


def test_lookup2(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=lookup2_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.kernel
def unary_kernel(foos: wp.array[wp.uint32]):
    i = wp.tid()

    foos[i] = wp.uint32(-1)
    x = -foos[i]
    foos[i] = x


def test_unary(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=unary_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.kernel
def rvalue_kernel(foos: wp.array[wp.uint32]):
    i = wp.tid()

    if foos[i] < wp.uint32(1):
        foos[i] = wp.uint32(1)


def test_rvalue(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=rvalue_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


# Tests, among other things, that assigning a reference to a new variable does
# not create a reference
@wp.kernel
def intermediate_kernel(foos: wp.array[wp.uint32]):
    i = wp.tid()

    x = foos[i]
    x = x + wp.uint32(1)
    foos[i] = x


def test_intermediate(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=intermediate_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.kernel
def array_kernel(foos: wp.array[wp.uint32]):
    i = wp.tid()
    foos[i] = wp.uint32(1)


def test_array_assign(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=array_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.func
def increment(arg: wp.uint32):
    return arg + wp.uint32(1)


@wp.kernel
def array_call_kernel(foos: wp.array[wp.uint32]):
    i = wp.tid()
    foos[i] = increment(foos[i])


def test_array_call_assign(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=array_call_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.struct
class Foo:
    field: wp.uint32


@wp.kernel
def array_struct_kernel(foos: wp.array[Foo]):
    i = wp.tid()
    foos[i].field = wp.uint32(1)


def test_array_struct_assign(test, device):
    foos = wp.zeros((10,), dtype=Foo, device=device)

    wp.launch(kernel=array_struct_kernel, dim=(10,), inputs=[foos], device=device)

    expected = Foo()
    expected.field = 1

    test.assertEqual(expected.field, 1)

    for f in foos.list():
        test.assertEqual(f.field, 1)


@wp.struct
class Bar:
    field: wp.uint32


@wp.struct
class Baz:
    bar: Bar


@wp.kernel
def array_struct_struct_kernel(foos: wp.array[Baz]):
    i = wp.tid()
    foos[i].bar.field = wp.uint32(1)


def test_array_struct_struct_assign(test, device):
    foos = wp.zeros((10,), dtype=Baz, device=device)

    wp.launch(kernel=array_struct_struct_kernel, dim=(10,), inputs=[foos], device=device)

    expected = Baz()
    expected.bar.field = 1

    test.assertEqual(expected.bar.field, 1)

    for f in foos.list():
        test.assertEqual(f.bar.field, 1)


@wp.struct
class S:
    a: wp.uint32
    b: wp.float32


@wp.struct
class F:
    x: wp.float32
    s: S
    y: wp.int32


@wp.kernel
def complex_kernel(foos: wp.array[F]):
    i = wp.tid()
    foos[i].x += wp.float32(1.0)
    foos[i].y = wp.int32(2)
    foos[i].s.b += wp.float32(3.0)
    foos[i].s.a = wp.uint32(foos[i].y)


def test_complex(test, device):
    foos = wp.zeros((10,), dtype=F, device=device)

    wp.launch(kernel=complex_kernel, dim=(10,), inputs=[foos], device=device)

    expected = F()
    expected.x = 1.0
    expected.y = 2
    expected.s.b = 3.0
    expected.s.a = expected.y
    for f in foos.list():
        test.assertEqual(f.x, expected.x)
        test.assertEqual(f.y, expected.y)
        test.assertEqual(f.s.a, expected.s.a)
        test.assertEqual(f.s.b, expected.s.b)


@wp.struct
class Svec:
    a: wp.uint32
    b: wp.vec2f


@wp.struct
class Fvec:
    x: wp.vec2f
    s: Svec
    y: wp.int32


@wp.kernel
def swizzle_kernel(foos: wp.array[Fvec]):
    i = wp.tid()

    foos[i].x += wp.vec2f(1.0, 2.0)
    foos[i].y = wp.int32(3)
    foos[i].s.b = wp.vec2f(4.0, 5.0)
    foos[i].s.b.y = wp.float32(6.0)
    foos[i].s.b.x = foos[i].x.y
    foos[i].s.a = wp.uint32(foos[i].y)


def test_swizzle(test, device):
    foos = wp.zeros((10,), dtype=Fvec, device=device)

    wp.launch(kernel=swizzle_kernel, dim=(10,), inputs=[foos], device=device)

    expected = Fvec()
    expected.x = wp.vec2f(1.0, 2.0)
    expected.y = 3
    expected.s.b = wp.vec2f(4.0, 5.0)
    expected.s.b.y = 6.0
    expected.s.b.x = expected.x.y
    expected.s.a = expected.y

    for f in foos.list():
        test.assertEqual(f.x, expected.x)
        test.assertEqual(f.y, expected.y)
        test.assertEqual(f.s.a, expected.s.a)
        test.assertEqual(f.s.b, expected.s.b)


@wp.kernel
def slice_kernel(a: wp.array2d[wp.vec3], b: wp.array2d[wp.vec3], c: wp.array2d[wp.vec3]):
    tid = wp.tid()
    c[tid][0] = a[tid][0] + b[tid][0]


def test_slice(test, device):
    a = wp.full((1, 1), value=1.0, dtype=wp.vec3, requires_grad=True, device=device)
    b = wp.full((1, 1), value=1.0, dtype=wp.vec3, requires_grad=True, device=device)
    c = wp.zeros((1, 1), dtype=wp.vec3, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=slice_kernel, dim=1, inputs=[a, b], outputs=[c], device=device)

    c.grad = wp.full((1, 1), value=1.0, dtype=wp.vec3, device=device)
    tape.backward()

    x = a.grad.list()[0]
    y = b.grad.list()[0]

    expected = wp.vec3(1.0)
    test.assertEqual(x, expected)
    test.assertEqual(y, expected)


@wp.kernel
def vec_component_iadd_kernel(a: wp.array[wp.vec3], b: wp.array[float], out: wp.array[float]):
    w = a[0]
    w.y += b[0]
    w.y += b[0]
    w.y += b[0]
    out[0] = w.y


@wp.kernel
def vec_subscript_iadd_kernel(a: wp.array[wp.vec3], b: wp.array[float], out: wp.array[float]):
    w = a[0]
    w[1] += b[0]
    w[1] += b[0]
    w[1] += b[0]
    out[0] = w[1]


def test_vec_component_iadd_matches_subscript(test, device):
    """``v.y += x`` and ``v[1] += x`` have the same gradient."""
    for kernel in (vec_component_iadd_kernel, vec_subscript_iadd_kernel):
        a = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, requires_grad=True, device=device)
        b = wp.array([0.5], dtype=float, requires_grad=True, device=device)
        out = wp.zeros(1, dtype=float, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[a, b, out], device=device)
        tape.backward(grads={out: wp.ones(1, dtype=float, device=device)})

        assert_np_equal(out.numpy(), np.array([3.5]))
        assert_np_equal(a.grad.numpy(), np.array([[0.0, 1.0, 0.0]]))
        assert_np_equal(b.grad.numpy(), np.array([3.0]))


@wp.kernel
def vec_component_mixed_kernel(x: wp.array[float], out: wp.array[float]):
    w = wp.vec3(x[0], x[1], x[2])
    w.x += x[3]
    w.y -= x[4]
    w.z += x[5]
    w.x -= x[4]
    w.y += w.x * x[3]
    out[0] = wp.length_sq(w)


@wp.kernel
def quat_component_kernel(x: wp.array[float], out: wp.array[float]):
    q = wp.quat(x[0], x[1], x[2], x[3])
    q.y += x[4]
    q.w -= x[5]
    q.y += x[4]
    q.w -= x[5]
    out[0] = wp.dot(q, wp.quat(1.0, 2.0, 3.0, 4.0))


@wp.kernel
def transform_translation_kernel(x: wp.array[float], out: wp.array[float]):
    t = wp.transform(wp.vec3(x[0], x[1], x[2]), wp.quat(x[3], x[4], x[5], x[6]))
    t.p += wp.vec3(x[7], x[8], x[9])
    t.p -= wp.vec3(x[9], x[7], x[8])
    t.p += wp.vec3(x[7], x[8], x[9])
    out[0] = wp.dot(t.p, t.p) + t.q.y


@wp.kernel
def transform_rotation_kernel(x: wp.array[float], out: wp.array[float]):
    t = wp.transform(wp.vec3(x[0], x[1], x[2]), wp.quat(x[3], x[4], x[5], x[6]))
    t.q += wp.quat(x[7], x[8], x[9], x[10])
    t.q -= wp.quat(x[10], x[7], x[8], x[9])
    t.q += wp.quat(x[7], x[8], x[9], x[10])
    out[0] = wp.dot(t.q, t.q) + t.p.y


@wp.func
def component_iadd_value_param(v: wp.vec3, b: float) -> float:
    v.y += b
    v.y += b
    return v.y


@wp.kernel
def func_value_param_component_kernel(x: wp.array[float], out: wp.array[float]):
    out[0] = component_iadd_value_param(wp.vec3(x[0], x[1], x[2]), x[3])


def test_component_augassign_finite_differences(test, device):
    """Repeated ``+=`` and ``-=`` on local components agree with central differences."""

    def evaluate(kernel, values):
        x = wp.array(values, dtype=float, device=device)
        out = wp.zeros(1, dtype=float, device=device)
        wp.launch(kernel, dim=1, inputs=[x, out], device=device)
        return out.numpy()[0]

    eps = 1e-2
    x_values = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 0.4, 0.6, 0.8, 0.5], dtype=np.float32)

    for kernel in (
        vec_component_mixed_kernel,
        quat_component_kernel,
        transform_translation_kernel,
        transform_rotation_kernel,
        func_value_param_component_kernel,
    ):
        x = wp.array(x_values, dtype=float, requires_grad=True, device=device)
        out = wp.zeros(1, dtype=float, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[x, out], device=device)
        tape.backward(grads={out: wp.ones(1, dtype=float, device=device)})

        expected = np.zeros_like(x_values)
        for i in range(len(x_values)):
            step = np.zeros_like(x_values)
            step[i] = eps
            expected[i] = (evaluate(kernel, x_values + step) - evaluate(kernel, x_values - step)) / (2.0 * eps)

        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-2, err_msg=kernel.key)


@wp.kernel
def transform_component_array_rhs_kernel(u: wp.array[wp.vec3], d: wp.array[wp.quat], out: wp.array[float]):
    t = wp.transform_identity()
    t.p += u[0]
    t.p += u[0]
    t.q += d[0]
    t.q -= d[0]
    t.q += d[0]
    out[0] = t.p.y + 2.0 * t.q.z


def test_transform_component_array_rhs(test, device):
    """Transform component updates accumulate into a right-hand side read from an array."""
    u = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, requires_grad=True, device=device)
    d = wp.array([[0.1, 0.2, 0.3, 0.4]], dtype=wp.quat, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(transform_component_array_rhs_kernel, dim=1, inputs=[u, d, out], device=device)
    tape.backward(grads={out: wp.ones(1, dtype=float, device=device)})

    assert_np_equal(u.grad.numpy(), np.array([[0.0, 2.0, 0.0]]))
    assert_np_equal(d.grad.numpy(), np.array([[0.0, 0.0, 2.0, 0.0]]))


@wp.func
def mutate_component_and_return_one(v: wp.ref[wp.vec3]) -> float:
    v.y = v.y + 10.0
    return 1.0


@wp.kernel(enable_backward=False)
def component_iadd_rhs_mutates_target_kernel(out: wp.array[float]):
    w = wp.vec3(1.0, 2.0, 3.0)
    w.y += mutate_component_and_return_one(w)
    out[0] = w.y


def test_component_iadd_rhs_mutates_target(test, device):
    """``v.y += f(v)`` adds to the value loaded before ``f`` writes through its reference."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(component_iadd_rhs_mutates_target_kernel, dim=1, inputs=[out], device=device)
    assert_np_equal(out.numpy(), np.array([3.0]))


devices = get_test_devices()


class TestLValue(unittest.TestCase):
    def test_swizzle_error_invalid_attribute(self):
        v = wp.vec3(1, 2, 3)
        with self.assertRaisesRegex(AttributeError, r"'vec3f' object has no attribute 'foo'$"):
            v.foo  # noqa: B018

        try:
            v.bar = 123
        except AttributeError:
            self.fail()

    def test_swizzle_error_invalid_component(self):
        v = wp.vec3(1, 2, 3)
        for name in ("xy", "yz", "zw", "wx", "xx", "xyz", "xyzw", "foo"):
            with self.assertRaisesRegex(AttributeError, rf"no attribute '{name}'$"):
                getattr(v, name)

        v2 = wp.vec2(1, 2)
        for name in ("z", "w"):
            with self.assertRaisesRegex(AttributeError, rf"no attribute '{name}'$"):
                getattr(v2, name)
        self.assertFalse(hasattr(v2, "w"))

        v3 = wp.vec3(1, 2, 3)
        for name in ("xy", "yz", "zw", "wx", "xx", "xyz", "xyzw"):
            with self.assertRaisesRegex(AttributeError, rf"no attribute '{name}'$"):
                setattr(v3, name, 9)
        v3.foo = 9
        self.assertEqual(v3.foo, 9)
        self.assertEqual(list(v3), [1.0, 2.0, 3.0])

        v4 = wp.vec2(1, 2)
        for name in ("z", "w"):
            with self.assertRaisesRegex(AttributeError, rf"no attribute '{name}'$"):
                setattr(v4, name, 9)
        self.assertEqual(list(v4), [1.0, 2.0])
        self.assertFalse(hasattr(v4, "w"))


add_function_test(TestLValue, "test_rmw_array", test_rmw_array, devices=devices)
add_function_test(TestLValue, "test_rmw_array_struct", test_rmw_array_struct, devices=devices)
add_function_test(TestLValue, "test_lookup", test_lookup, devices=devices)
add_function_test(TestLValue, "test_lookup2", test_lookup2, devices=devices)
add_function_test(TestLValue, "test_grad", test_grad, devices=devices)
add_function_test(TestLValue, "test_unary", test_unary, devices=devices)
add_function_test(TestLValue, "test_rvalue", test_rvalue, devices=devices)
add_function_test(TestLValue, "test_intermediate", test_intermediate, devices=devices)
add_function_test(TestLValue, "test_array_assign", test_array_assign, devices=devices)
add_function_test(TestLValue, "test_array_struct_assign", test_array_struct_assign, devices=devices)
add_function_test(TestLValue, "test_array_struct_struct_assign", test_array_struct_struct_assign, devices=devices)
add_function_test(TestLValue, "test_complex", test_complex, devices=devices)
add_function_test(TestLValue, "test_swizzle", test_swizzle, devices=devices)
add_function_test(TestLValue, "test_slice", test_slice, devices=devices)
add_function_test(
    TestLValue, "test_vec_component_iadd_matches_subscript", test_vec_component_iadd_matches_subscript, devices=devices
)
add_function_test(
    TestLValue,
    "test_component_augassign_finite_differences",
    test_component_augassign_finite_differences,
    devices=devices,
)
add_function_test(TestLValue, "test_transform_component_array_rhs", test_transform_component_array_rhs, devices=devices)
add_function_test(
    TestLValue, "test_component_iadd_rhs_mutates_target", test_component_iadd_rhs_mutates_target, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
