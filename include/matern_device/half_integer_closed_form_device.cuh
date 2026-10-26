/* half_integer_closed_form_device.cuh -- device half-integer closed form, M = e^-Z P(Z),
 * with the order tested per thread (see the accompanying paper, half-integer orders).
 * Host counterpart: half_integer_closed_form.h. */
#ifndef MATERN_HALF_INTEGER_CLOSED_FORM_DEVICE_CUH
#define MATERN_HALF_INTEGER_CLOSED_FORM_DEVICE_CUH

#include "dd_arithmetic_device.cuh"

/* Highest polynomial degree in the device table (nu <= 32.5), and the table size. */
#define MATERN_HALF_INTEGER_MAX_DEGREE_DEVICE 32
#define MATERN_HALF_INTEGER_TABLE_SIZE_DEVICE \
    (((MATERN_HALF_INTEGER_MAX_DEGREE_DEVICE)+1)*((MATERN_HALF_INTEGER_MAX_DEGREE_DEVICE)+2)/2)

/* Triangular, row p at offset p(p+1)/2: c_0 .. c_p, built by the ratio recurrence in x87
 * long double and rounded once to double (tools/half_integer_table_device.c; make audit
 * compares the two); statically initialised.  static: a file-scope __constant__ with
 * external linkage would be a duplicate symbol across linked translation units. */
static __device__ __constant__ double matern_half_integer_coefficients_device[MATERN_HALF_INTEGER_TABLE_SIZE_DEVICE] = {
    /* p = 0 */
    0x1p+0,
    /* p = 1 */
    0x1p+0, 0x1p+0,
    /* p = 2 */
    0x1p+0, 0x1p+0, 0x1.5555555555555p-2,
    /* p = 3 */
    0x1p+0, 0x1p+0, 0x1.999999999999ap-2, 0x1.1111111111111p-4,
    /* p = 4 */
    0x1p+0, 0x1p+0, 0x1.b6db6db6db6dbp-2, 0x1.8618618618618p-4,
    0x1.3813813813814p-7,
    /* p = 5 */
    0x1p+0, 0x1p+0, 0x1.c71c71c71c71cp-2, 0x1.c71c71c71c71cp-4,
    0x1.041041041041p-6, 0x1.1566abc011567p-10,
    /* p = 6 */
    0x1p+0, 0x1p+0, 0x1.d1745d1745d17p-2, 0x1.f07c1f07c1f08p-4,
    0x1.4afd6a052bf5bp-6, 0x1.08cabb37565e2p-9, 0x1.937e11175f095p-14,
    /* p = 7 */
    0x1p+0, 0x1p+0, 0x1.d89d89d89d89ep-2, 0x1.069069069069p-3,
    0x1.7de952f2466a4p-6, 0x1.6ea28d118b474p-9, 0x1.b287c3a303e2bp-13, 0x1.f09b28ba4d955p-18,
    /* p = 8 */
    0x1p+0, 0x1p+0, 0x1.ddddddddddddep-2, 0x1.1111111111111p-3,
    0x1.a41a41a41a41ap-6, 0x1.c01c01c01c01cp-9, 0x1.45e5d2ba42eap-12, 0x1.29f6b20961cp-16,
    0x1.08db48ebe51c7p-21,
    /* p = 9 */
    0x1p+0, 0x1p+0, 0x1.e1e1e1e1e1e1ep-2, 0x1.1919191919192p-3,
    0x1.c1c1c1c1c1c1cp-6, 0x1.010101010101p-8, 0x1.a5c001a5c001ap-12, 0x1.e20001e20001ep-16,
    0x1.5e8ba44745d2dp-20, 0x1.f28db670be53bp-26,
    /* p = 10 */
    0x1p+0, 0x1p+0, 0x1.e50d79435e50dp-2, 0x1.1f7047dc11f7p-3,
    0x1.d96da388960f5p-6, 0x1.1c0e9551f3a2dp-8, 0x1.f8fd7b3c5bcc1p-12, 0x1.49ca1c3c50d8ep-15,
    0x1.306bcb4b5e52p-19, 0x1.68cb9b9bb2285p-24, 0x1.a3d5a71b92cd3p-30,
    /* p = 11 */
    0x1p+0, 0x1p+0, 0x1.e79e79e79e79ep-2, 0x1.2492492492492p-3,
    0x1.ecc07b301eccp-6, 0x1.3299e6401329ap-8, 0x1.2090d8b4c6bdcp-11, 0x1.9c3ca34b650f1p-15,
    0x1.b7b825a5c1213p-19, 0x1.4f06351093257p-23, 0x1.49deba27f3581p-28, 0x1.3fdfbc45c52eap-34,
    /* p = 12 */
    0x1p+0, 0x1p+0, 0x1.e9bd37a6f4deap-2, 0x1.28cfc4a33f129p-3,
    0x1.fcd1e360fe68fp-6, 0x1.45a50c670938fp-8, 0x1.3fee80f4f29acp-11, 0x1.e783d0b234bbp-15,
    0x1.1ec6024ab59b3p-18, 0x1.fdd1cb2f7bbe9p-23, 0x1.4648d3f56deaap-27, 0x1.0f328eed3d6fep-32,
    0x1.bd0ac3296b624p-39,
    /* p = 13 */
    0x1p+0, 0x1p+0, 0x1.eb851eb851eb8p-2, 0x1.2c5f92c5f92c6p-3,
    0x1.0531b747fa105p-5, 0x1.55ed4d05c9afp-8, 0x1.5b5ab7e55f2bap-11, 0x1.15e22cb77f562p-14,
    0x1.5f02bf38a0d89p-18, 0x1.5aad6134c4c94p-22, 0x1.05070ff78b221p-26, 0x1.1cc1e2df80824p-31,
    0x1.94fcfe65b1145p-37, 0x1.1cd3b01a822a6p-43,
    /* p = 14 */
    0x1p+0, 0x1p+0, 0x1.ed097b425ed09p-2, 0x1.2f684bda12f68p-3,
    0x1.0aff9ee8dd7ccp-5, 0x1.63ff7e8bd1fbbp-8, 0x1.7379e838db1cep-11, 0x1.34c26a92ff434p-14,
    0x1.9bade36ea9afp-18, 0x1.b71fe1871b655p-22, 0x1.71ca01495a704p-26, 0x1.de1c7370f10d5p-31,
    0x1.c1fca8e2c4c13p-36, 0x1.14ea408b8cc5bp-41, 0x1.519297d390c9fp-48,
    /* p = 15 */
    0x1p+0, 0x1p+0, 0x1.ee58469ee5847p-2, 0x1.3205e293205e3p-3,
    0x1.10053b2d721adp-5, 0x1.704618fa929a7p-8, 0x1.88d34dd813d7fp-11, 0x1.50b51e26ec6ffp-14,
    0x1.d4766cbbb842cp-18, 0x1.08fd2b67995c1p-21, 0x1.e48d1c2b270ebp-26, 0x1.6066a01f62394p-30,
    0x1.8bad9d54a42e5p-35, 0x1.44a8b59ac85a8p-40, 0x1.5d368b681a32p-46, 0x1.747e72912d023p-53,
    /* p = 16 */
    0x1p+0, 0x1p+0, 0x1.ef7bdef7bdef8p-2, 0x1.344d1344d1345p-3,
    0x1.1468698d257e5p-5, 0x1.7b12d9e624ca8p-8, 0x1.9bd545393e181p-11, 0x1.6a0d12a87a1adp-14,
    0x1.04ad40a2436f7p-17, 0x1.34f3431f20841p-21, 0x1.2ce4030a47bccp-25, 0x1.dd72444b8d55dp-30,
    0x1.2f23fa999ed0ep-34, 0x1.2a7a12505d5b7p-39, 0x1.aee244c8b4f78p-45, 0x1.988aaf33dec89p-51,
    0x1.808286c768445p-58,
    /* p = 17 */
    0x1p+0, 0x1p+0, 0x1.f07c1f07c1f08p-2, 0x1.364d9364d9365p-3,
    0x1.1846118461184p-5, 0x1.84a56d9fac2f5p-8, 0x1.acd9da114b3d1p-11, 0x1.811736c129704p-14,
    0x1.1d409a5631a88p-17, 0x1.5f146f2f020a7p-21, 0x1.678177fefbf22p-25, 0x1.310900f76168ap-29,
    0x1.a865920a44be1p-34, 0x1.dad97a3a08e3p-39, 0x1.9d797322dc336p-44, 0x1.089fa5d8ded93p-49,
    0x1.bdae9095ad37ep-56, 0x1.74dba97b8be52p-63,
    /* p = 18 */
    0x1p+0, 0x1p+0, 0x1.f15f15f15f15fp-2, 0x1.3813813813814p-3,
    0x1.1bb4a4046ed29p-5, 0x1.8d3018d3018d3p-8, 0x1.bc2ac66d57118p-11, 0x1.96187ae7a01p-14,
    0x1.341292342126ap-17, 0x1.873406dc9bdfcp-21, 0x1.a1488fda3fddap-25, 0x1.7583303423af2p-29,
    0x1.16e3a91cb0d4bp-33, 0x1.573f950f9e8f9p-38, 0x1.551dcbc98f43fp-43, 0x1.089fa5d8ded93p-48,
    0x1.2e6d4fd347d3ap-54, 0x1.c76b63200bc94p-61, 0x1.54e6174c62a5ap-68,
    /* p = 19 */
    0x1p+0, 0x1p+0, 0x1.f22983759f22ap-2, 0x1.39a85c40939a8p-3,
    0x1.1ec5d0b011ec6p-5, 0x1.94db0880194dbp-8, 0x1.ca04b97479b79p-11, 0x1.a94d87a3034fp-14,
    0x1.494447fa1313ep-17, 0x1.ad4428bd7d602p-21, 0x1.d9ac5043d9cb2p-25, 0x1.baea446757499p-29,
    0x1.5df5263eb06cbp-33, 0x1.cfd961a66791ap-38, 0x1.fce8a5a67a683p-43, 0x1.c45d0505c223bp-48,
    0x1.3ab0037e70cafp-53, 0x1.431a065346a1dp-59, 0x1.b5a3ed7a474a4p-66, 0x1.26d4d5e132b8cp-73,
    /* p = 20 */
    0x1p+0, 0x1p+0, 0x1.f2df2df2df2dfp-2, 0x1.3b13b13b13b14p-3,
    0x1.2187b79e12188p-5, 0x1.9bc6b58019bc7p-8, 0x1.d699f4001d69ap-11, 0x1.baeb40001baebp-14,
    0x1.5cf768ba445b2p-17, 0x1.d149e0f85b242p-21, 0x1.0829f1afaf9dap-24, 0x1.0028ac4d45721p-28,
    0x1.a7fcb3421aab1p-33, 0x1.2a30510f8bb9p-37, 0x1.6168abec7fabep-42, 0x1.5bf8c8c95a42ep-47,
    0x1.1660a0a115024p-52, 0x1.5d561ee842856p-58, 0x1.4405ad5cfea0dp-64, 0x1.8ce36eaf26bdap-71,
    0x1.e3d36c1c38faap-79,
    /* p = 21 */
    0x1p+0, 0x1p+0, 0x1.f3831f3831f38p-2, 0x1.3c5b93c5b93c6p-3,
    0x1.2405c37b6fe8fp-5, 0x1.a20da3f977c97p-8, 0x1.e214579c371bp-11, 0x1.cb1f906403b83p-14,
    0x1.6f4c7383362dp-17, 0x1.f3558748f7a9ep-21, 0x1.22858b386adbep-24, 0x1.22858b386adbep-28,
    0x1.f3d27be7f1a65p-33, 0x1.71197711ae82bp-37, 0x1.d17720e0124c9p-42, 0x1.f07f12001384dp-47,
    0x1.b95481c72dcb7p-52, 0x1.3f83d795fe674p-57, 0x1.6b897dd27b56cp-63, 0x1.3222f0b13f718p-69,
    0x1.54be438da1e28p-76, 0x1.799ec4c4db4dp-84,
    /* p = 22 */
    0x1p+0, 0x1p+0, 0x1.f417d05f417dp-2, 0x1.3d84f613d84f6p-3,
    0x1.264947f96ad2fp-5, 0x1.a7c5af52ae4e8p-8, 0x1.ec9704aee07e4p-11, 0x1.da124da660b71p-14,
    0x1.8061de1832bdfp-17, 0x1.09beebb83c5d6p-20, 0x1.3bdb9bd3a9c26p-24, 0x1.444da15cbeb9ep-28,
    0x1.20450135fede2p-32, 0x1.bb7dda7a74697p-37, 0x1.264c6ff2e1e54p-41, 0x1.4ed8ace248b2ep-46,
    0x1.434cca3b95a3ep-51, 0x1.04d03be29f70ep-56, 0x1.57753f0f0ada2p-62, 0x1.63f9000f972aap-68,
    0x1.1163127a90c49p-74, 0x1.15b9fa642149cp-81, 0x1.19050980a3336p-89,
    /* p = 23 */
    0x1p+0, 0x1p+0, 0x1.f49f49f49f49fp-2, 0x1.3e93e93e93e94p-3,
    0x1.2859f6bd304a1p-5, 0x1.ad0103a908f3dp-8, 0x1.f63fa061fdfe3p-11, 0x1.e7e60983c3893p-14,
    0x1.9053b909a701p-17, 0x1.18ee66e755c1dp-20, 0x1.5427941f05964p-24, 0x1.6555898043eep-28,
    0x1.46b49af8eda67p-32, 0x1.042e870028fdp-36, 0x1.686cc8500395ep-41, 0x1.b08289f99de71p-46,
    0x1.be763bd860ee8p-51, 0x1.882fa4015626ep-56, 0x1.208134f827d5fp-61, 0x1.5b12c67599c99p-67,
    0x1.4913b2affa1c1p-73, 0x1.cee0a1518cd04p-80, 0x1.aee5971a93d74p-87, 0x1.8fac24452c5fdp-95,
    /* p = 24 */
    0x1p+0, 0x1p+0, 0x1.f51b3bea3677dp-2, 0x1.3f8bcd29c245p-3,
    0x1.2a3e36f3c6628p-5, 0x1.b1cedb91208f5p-8, 0x1.ff275610363dbp-11, 0x1.f4b8d1b1d717ep-14,
    0x1.9f3b9b2954b62p-17, 0x1.2746d4bcae046p-20, 0x1.6b6ade7211407p-24, 0x1.8580033bc8ff4p-28,
    0x1.6cefde20f3ac4p-32, 0x1.2b6f7b2ebacfp-36, 0x1.ae3596dbb17dp-41, 0x1.0def6db705f41p-45,
    0x1.26798ef6350a4p-50, 0x1.15271d23f5af5p-55, 0x1.bd08ab949499p-61, 0x1.2bcff23e5eb7fp-66,
    0x1.4ad3cd829d7b8p-72, 0x1.20113e44743adp-78, 0x1.7473629bbd109p-85, 0x1.3ee3382c4c418p-92,
    0x1.101e0dcd0ddf3p-100,
    /* p = 25 */
    0x1p+0, 0x1p+0, 0x1.f58d0fac687d6p-2, 0x1.406f74ae26502p-3,
    0x1.2bfb67c929507p-5, 0x1.b63c120d6232cp-8, 0x1.03b1d1cf0ac8cp-10, 0x1.00526b57fd587p-13,
    0x1.ad30b3bd01823p-17, 0x1.34d5d6ab389ebp-20, 0x1.81aae19d9f5e8p-24, 0x1.a4ba6a7d680ap-28,
    0x1.92bf93e342c5cp-32, 0x1.532810bf60a69p-36, 0x1.f6d79267d96d8p-41, 0x1.47c74e5e3e161p-45,
    0x1.769aa2b4d93ddp-50, 0x1.754ece6f21206p-55, 0x1.41c6150fa06eap-60, 0x1.da316267def3ep-66,
    0x1.25b1947bca7ffp-71, 0x1.2a5b007dc1861p-77, 0x1.dedd7ec83c39dp-84, 0x1.1d88d785b49b5p-90,
    0x1.c3370f0c2d34bp-98, 0x1.636b2c25f2c57p-106,
    /* p = 26 */
    0x1p+0, 0x1p+0, 0x1.f5f5f5f5f5f5fp-2, 0x1.4141414141414p-3,
    0x1.2d961376baa5cp-5, 0x1.ba53940378265p-8, 0x1.07837e4e5278ep-10, 0x1.05e07d857ce44p-13,
    0x1.ba47f06fab1b1p-17, 0x1.41a8aedcd9881p-20, 0x1.96ef995a1464bp-24, 0x1.c2fa03b6f2076p-28,
    0x1.b7fa291662c28p-32, 0x1.7b0ea94e5cf66p-36, 0x1.20ce37da2e729p-40, 0x1.8520a17c33cd9p-45,
    0x1.cebf0539c1102p-50, 0x1.e3eae5ac80466p-55, 0x1.ba705cf57c982p-60, 0x1.5ea9e5d009fb6p-65,
    0x1.dc0d6fd7b6aa9p-71, 0x1.1007ad9fd6185p-76, 0x1.fe8e090df4eadp-83, 0x1.7ad8b2de939a4p-89,
    0x1.a209993c38f0dp-96, 0x1.31c3baafb49e9p-103, 0x1.be03fb2a9a10ep-112,
    /* p = 27 */
    0x1p+0, 0x1p+0, 0x1.f656f1826a43ap-2, 0x1.4203385a29dc9p-3,
    0x1.2f1216eb72b18p-5, 0x1.be1eb94a320b6p-8, 0x1.0b10a6a652342p-10, 0x1.0b10a6a652342p-13,
    0x1.c6942bffdd9f8p-17, 0x1.4dcc0793120acp-20, 0x1.ab42990e2b92ep-24, 0x1.e039ed9dadcd5p-28,
    0x1.dc80ebc6218a4p-32, 0x1.a2e77db0fe8a6p-36, 0x1.46f31d6aec211p-40, 0x1.c55ec5ce4ad89p-45,
    0x1.16ff3ea655719p-49, 0x1.300bdba7cb484p-54, 0x1.242d388ecb08bp-59, 0x1.ec16444ecf371p-65,
    0x1.67ed2d92e0b4bp-70, 0x1.c3ac251cb5978p-76, 0x1.ddcd6062129cp-82, 0x1.9f7aefa33192dp-88,
    0x1.1debffc0221d7p-94, 0x1.24c8b3f7f1c82p-101, 0x1.8da087c42b29ap-109, 0x1.0d4adb45318cap-117,
    /* p = 28 */
    0x1p+0, 0x1p+0, 0x1.f6b0df6b0df6bp-2, 0x1.42b7142b7142bp-3,
    0x1.3072c0ea30e33p-5, 0x1.c1a58b327f576p-8, 0x1.0e60323c7b6bcp-10, 0x1.0febb7a4b0cap-13,
    0x1.d2265f637835bp-17, 0x1.594bd4e16286ap-20, 0x1.beae5a31b930ep-24, 0x1.fc79922e7bca5p-28,
    0x1.001ee520e53a5p-31, 0x1.ca827f55be2b8p-36, 0x1.6d96d41823f67p-40, 0x1.03f97a60cf9e3p-44,
    0x1.49b94a0a64fa9p-49, 0x1.7464d823d56f5p-54, 0x1.757472a5c259cp-59, 0x1.4b0a53305cda9p-64,
    0x1.01acb4fd83c73p-69, 0x1.5d050567a397fp-75, 0x1.9621c520296bp-81, 0x1.8edc6b2c06a16p-87,
    0x1.424fda7b7ee4bp-93, 0x1.9c8f2c2331c89p-100, 0x1.891bcc0aaaac1p-107, 0x1.f0f786b090f7dp-115,
    0x1.395bc2a44c468p-123,
    /* p = 29 */
    0x1p+0, 0x1p+0, 0x1.f7047dc11f704p-2, 0x1.435e50d79435ep-3,
    0x1.31baeaafe4901p-5, 0x1.c4eefcdea7f1ep-8, 0x1.117824c065669p-10, 0x1.147976bce24b9p-13,
    0x1.dd0dd1e68b7d5p-17, 0x1.6433476475ce4p-20, 0x1.d13dc083369a7p-24, 0x1.0bddbc6a93cd4p-27,
    0x1.1190c06cd84efp-31, 0x1.f1b972a57f46dp-36, 0x1.947ff525c9c62p-40, 0x1.262e83be6433p-44,
    0x1.7f1ecf4b4ce92p-49, 0x1.be6fe9aa35c46p-54, 0x1.d095c34b25389p-59, 0x1.ae5a38f2124f5p-64,
    0x1.611c1477da82cp-69, 0x1.fdc098bc6b908p-75, 0x1.40a165bb13399p-80, 0x1.5af661b7e355cp-86,
    0x1.3d3908e2a3f6ap-92, 0x1.ddb340948a7c7p-99, 0x1.1d0faa6a8abbap-105, 0x1.fac6844ba14d9p-113,
    0x1.2aed6de377ebbp-120, 0x1.5fd74f512d38cp-129,
    /* p = 30 */
    0x1p+0, 0x1p+0, 0x1.f75270d0456c8p-2, 0x1.43fa36f5e02e5p-3,
    0x1.32ed0ba591037p-5, 0x1.c801189e3688cp-8, 0x1.145dc158210d1p-10, 0x1.18c0c469c82dcp-13,
    0x1.e7584673ffb4ep-17, 0x1.6e8cc9c6d634fp-20, 0x1.e2fbc49992b83p-24, 0x1.19023138c512bp-27,
    0x1.229111ceba5c8p-31, 0x1.0c37242149907p-35, 0x1.bb7e30d8e3764p-40, 0x1.4915a911b9d9ap-44,
    0x1.b6c78c17a2777p-49, 0x1.06cc645557b93p-53, 0x1.1a7dc67a2ab33p-58, 0x1.0fdf0fdb96cf2p-63,
    0x1.d2d2ae95c5c29p-69, 0x1.63acb5c7721a5p-74, 0x1.dd8c96f9e2998p-80, 0x1.17c113e7541f5p-85,
    0x1.1a4646d2597e3p-91, 0x1.e1bf9e6a68f2ap-98, 0x1.52cfe86157e02p-104, 0x1.79ef2a41147f4p-111,
    0x1.3a208d828f5fdp-118, 0x1.5a9f81aa8c8d1p-126, 0x1.7da878c0310ddp-135,
    /* p = 31 */
    0x1p+0, 0x1p+0, 0x1.f79b47582192ep-2, 0x1.448be405987b2p-3,
    0x1.340b493db7c75p-5, 0x1.cae124b7bec98p-8, 0x1.1715a78c3229cp-10, 0x1.1cc7baa3f47e3p-13,
    0x1.f1122530cb554p-17, 0x1.786204fbdc55ap-20, 0x1.f3f339d1ed9b4p-24, 0x1.25ad4eb82e7b7p-27,
    0x1.331cf9a429ea8p-31, 0x1.1f44d9c1ba819p-35, 0x1.e2692bffead51p-40, 0x1.6c7cf96c06c32p-44,
    0x1.f0530762141ap-49, 0x1.30a6325cf02f6p-53, 0x1.50fea32ac1a08p-58, 0x1.4f61dbf4a833fp-63,
    0x1.2b8110a154b88p-68, 0x1.de1eb99acbbeap-74, 0x1.533e1a1d3ada2p-79, 0x1.a8ca6c62ee6c7p-85,
    0x1.d0ba723357fc2p-91, 0x1.b64fea5c9e6fap-97, 0x1.5debb53ec3acbp-103, 0x1.cccd0c2cba0a6p-110,
    0x1.e17d6ffc946bp-117, 0x1.7708e89f591eep-124, 0x1.83ea2f3075017p-132, 0x1.906d9c10fce8bp-141,
    /* p = 32 */
    0x1p+0, 0x1p+0, 0x1.f7df7df7df7dfp-2, 0x1.4514514514514p-3,
    0x1.351783d491385p-5, 0x1.cd93c1740c0fdp-8, 0x1.19a3eb311d0ep-10, 0x1.2093c75e760bdp-13,
    0x1.fa46a12379c3ep-17, 0x1.81bbe87c8d891p-20, 0x1.021753dfe78b3p-23, 0x1.31e2c238c692p-27,
    0x1.4333377ac8265p-31, 0x1.31fcfc7a4d656p-35, 0x1.048fb8290ab0bp-39, 0x1.9038e9b7df44bp-44,
    0x1.15b48d461d887p-48, 0x1.5c7e2ec66b55fp-53, 0x1.8b73f3b765d3ep-58, 0x1.956839b4660c3p-63,
    0x1.76c6c027565f7p-68, 0x1.37809fb64e709p-73, 0x1.cfa1ac3903a1ap-79, 0x1.332afad8eb9d7p-84,
    0x1.679c76d87dfdp-90, 0x1.703dea56851bap-96, 0x1.4563a62b2048ep-102, 0x1.e721e84c8a284p-109,
    0x1.2cee65fddcc2ep-115, 0x1.272a9128145d1p-122, 0x1.afc93dab45494p-130, 0x1.a37f05059b3cfp-138,
    0x1.96c8bf0d31af7p-147,
};

static __device__ __host__ inline int matern_half_integer_offset_device(int p){ return p*(p+1)/2; }


/* p for nu = p + 1/2 within the table, -1 otherwise, so the caller tests a sign.
 * The comparison t == r is exact, not a tolerance: nu - 1/2 is computed exactly in
 * the table's range, and lands on an integer only at a true half-integer order. */
static __device__ __host__ inline int matern_half_integer_degree_device(double nuc)
{
    double t = nuc - 0.5;
    double r = floor(t + 0.5);
    if (t != r) return -1;
    /* Range test before the cast: for very large nu, nuc - 0.5 is inexact enough
     * that t == r holds, and converting an out-of-range double to int is
     * undefined. */
    if (!(r >= 0.0) || r > (double)MATERN_HALF_INTEGER_MAX_DEGREE_DEVICE) return -1;
    return (int)r;
}


/* Direct form e^-Zh P(Zh); it loses bits where e^-Zh is subnormal (Zh > 708.396).  The
 * factor (1 - Zl + Zl*d/s) includes e^-Zl, because only -Zh is exponentiated. */
static __device__ inline double matern_half_integer_direct_form_device(double Zh, double Zl, int p)
{
    const double *c = matern_half_integer_coefficients_device + matern_half_integer_offset_device(p);
    double s = c[p], d = 0.0;
    /* simultaneous Horner: s ends as P(Zh) and d as P'(Zh), which supplies the
     * argument correction */
    for (int j = p - 1; j >= 0; j--){ d = fma(d, Zh, s); s = fma(s, Zh, c[j]); }
    return exp(-Zh) * s * (1.0 - Zl + Zl * (d / s));
}

/* exp(-Z + ln P(Z)) with a double-double exponent carrying -Zh-Zl exactly, so only the
 * polynomial is corrected for Zl. */
static __device__ inline double matern_half_integer_exp_log_device(double Zh, double Zl, int p)
{
    const double *c = matern_half_integer_coefficients_device + matern_half_integer_offset_device(p);
    double s = c[p], d = 0.0;
    for (int j = p - 1; j >= 0; j--){ d = fma(d, Zh, s); s = fma(s, Zh, c[j]); }
    const double pc = fma(d, Zl, s);           /* all c_j > 0, Zh >= 0 => pc >= 1 */
    double lh, ll; matern_dd_log_device(pc, &lh, &ll);
    double Eh, e1; matern_two_sum_device(lh, -Zh, &Eh, &e1);
    double El = (e1 + ll) - Zl;
    /* Second order in El, since dd_log's low word is not normalised (as on the CPU). */
    double eE = exp(Eh);
    eE = fma(eE, fma(0.5 * El, El, El), eE);
    return (eE > 0.0 && eE <= 1.0) ? eE : 0.0; /* M lies in (0, 1] */
}

#endif /* MATERN_HALF_INTEGER_CLOSED_FORM_DEVICE_CUH */
