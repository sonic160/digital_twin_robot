#ifndef CM__VISIBILITY_CONTROL_H_
#define CM__VISIBILITY_CONTROL_H_
#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define CM_EXPORT __attribute__ ((dllexport))
    #define CM_IMPORT __attribute__ ((dllimport))
  #else
    #define CM_EXPORT __declspec(dllexport)
    #define CM_IMPORT __declspec(dllimport)
  #endif
  #ifdef CM_BUILDING_LIBRARY
    #define CM_PUBLIC CM_EXPORT
  #else
    #define CM_PUBLIC CM_IMPORT
  #endif
  #define CM_PUBLIC_TYPE CM_PUBLIC
  #define CM_LOCAL
#else
  #define CM_EXPORT __attribute__ ((visibility("default")))
  #define CM_IMPORT
  #if __GNUC__ >= 4
    #define CM_PUBLIC __attribute__ ((visibility("default")))
    #define CM_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define CM_PUBLIC
    #define CM_LOCAL
  #endif
  #define CM_PUBLIC_TYPE
#endif
#endif  // CM__VISIBILITY_CONTROL_H_
// Generated 20-Mar-2024 15:51:48
// Copyright 2019-2020 The MathWorks, Inc.
