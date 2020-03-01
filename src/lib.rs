//! FFI bindings for libsvm
#![allow(non_snake_case)]

use std::os::raw::c_int;

pub const LIBSVM_VERSION: u32 = 324;
extern "C" {
    pub static mut libsvm_version: c_int;
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct svm_node {
    pub index: c_int,
    pub value: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct svm_problem {
    pub l: c_int,
    pub y: *mut f64,
    pub x: *mut *mut svm_node,
}

pub const C_SVC: u32 = 0;
pub const NU_SVC: u32 = 1;
pub const ONE_CLASS: u32 = 2;
pub const EPSILON_SVR: u32 = 3;
pub const NU_SVR: u32 = 4;
pub const LINEAR: u32 = 0;
pub const POLY: u32 = 1;
pub const RBF: u32 = 2;
pub const SIGMOID: u32 = 3;
pub const PRECOMPUTED: u32 = 4;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct svm_parameter {
    pub svm_type: c_int,
    pub kernel_type: c_int,
    pub degree: c_int,
    pub gamma: f64,
    pub coef0: f64,
    pub cache_size: f64,
    pub eps: f64,
    pub C: f64,
    pub nr_weight: c_int,
    pub weight_label: *mut c_int,
    pub weight: *mut f64,
    pub nu: f64,
    pub p: f64,
    pub shrinking: c_int,
    pub probability: c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct svm_model {
    pub param: svm_parameter,
    pub nr_class: c_int,
    pub l: c_int,
    pub SV: *mut *mut svm_node,
    pub sv_coef: *mut *mut f64,
    pub rho: *mut f64,
    pub probA: *mut f64,
    pub probB: *mut f64,
    pub sv_indices: *mut c_int,
    pub label: *mut c_int,
    pub nSV: *mut c_int,
    pub free_sv: c_int,
}

extern "C" {
    pub fn svm_train(prob: *const svm_problem, param: *const svm_parameter) -> *mut svm_model;

    pub fn svm_cross_validation(
        prob: *const svm_problem,
        param: *const svm_parameter,
        nr_fold: c_int,
        target: *mut f64,
    );

    pub fn svm_save_model(
        model_file_name: *const ::std::os::raw::c_char,
        model: *const svm_model,
    ) -> c_int;

    pub fn svm_load_model(model_file_name: *const ::std::os::raw::c_char) -> *mut svm_model;

    pub fn svm_get_svm_type(model: *const svm_model) -> c_int;

    pub fn svm_get_nr_class(model: *const svm_model) -> c_int;

    pub fn svm_get_labels(model: *const svm_model, label: *mut c_int);

    pub fn svm_get_sv_indices(model: *const svm_model, sv_indices: *mut c_int);

    pub fn svm_get_nr_sv(model: *const svm_model) -> c_int;

    pub fn svm_get_svr_probability(model: *const svm_model) -> f64;

    pub fn svm_predict_values(
        model: *const svm_model,
        x: *const svm_node,
        dec_values: *mut f64,
    ) -> f64;

    pub fn svm_predict(model: *const svm_model, x: *const svm_node) -> f64;

    pub fn svm_predict_probability(
        model: *const svm_model,
        x: *const svm_node,
        prob_estimates: *mut f64,
    ) -> f64;

    pub fn svm_free_model_content(model_ptr: *mut svm_model);

    pub fn svm_free_and_destroy_model(model_ptr_ptr: *mut *mut svm_model);

    pub fn svm_destroy_param(param: *mut svm_parameter);

    pub fn svm_check_parameter(
        prob: *const svm_problem,
        param: *const svm_parameter,
    ) -> *const ::std::os::raw::c_char;

    pub fn svm_check_probability_model(model: *const svm_model) -> c_int;

    pub fn svm_set_print_string_function(
        print_func: ::std::option::Option<
            unsafe extern "C" fn(arg1: *const ::std::os::raw::c_char),
        >,
    );
}
