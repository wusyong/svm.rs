fn main() {
    cc::Build::new()
        .include("libsvm")
        .file("libsvm/svm.cpp")
        .cpp(true)
        .flag("-std=c++11")
        .compile("libsvm");
}
