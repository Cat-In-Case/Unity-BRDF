Shader "Cook TOrreance"
{
    Properties
    {
        [KeywordEnum(Lit, Toon)] _Type("Rendrer Type", Float) = 0
        _MainTex ("Texture", 2D) = "white" {}
        [Toggle(TOGGLE_NORMAL)] _NORMALTOGGLE("Use Normal", Float) = 0
        _NormalTex("NormalMap",2D) = "bump" {}
        _NormalPower("Normal Power", Range(0, 30)) = 1
        _ShadowAdjust("shadowAdjust", Range(0, 1)) = 0

        [KeywordEnum(Specular, Metallic)] _SurfaceType("Rendrer Type", Float) = 0

        [KeywordEnum(None, Use)] _MAP("Use Combined Map", Float) = 0
        _MaskMap("Occlusion(R), Glossness(G), Metallic(B), ShadowMap(A)", 2D) = "white" {}

        [Toggle(TOGGLE_SM_COMBINE)] _COMBINETOGGLE("Add Metallic on Roughness (Substance)", Float) = 0
        [KeywordEnum(Smoothness Map, Roughness Map)] _RoughType("Map Type(S, R)", Float) = 0
        _SmoothnessMap("Smoothness Map", 2D) = "black" {}
        _Smoothness("Smoothness", Range(0, 1)) = 0
        _MetallicMap("Metallic Map", 2D) = "white" {}
        _MetallicMapIntensity("Metallic Map Intensity", Range(.001, 5)) = 1 
        _Metallic("_Metallic", Range(0, 1)) = 0
        _AOMAP("Ambient Occlusion", 2D)= "white" {}
        _AOIntensity("AO Intensity", Range(0, 1)) = 1

        [Toggle(TOGGLE_EMISSION)] _EMISSIONTOGGLE("Use Emisssion", Float) = 0
        [HDR] _Emission("Emission", Color) = (0, 0, 0, 0)
        _EmissionMap("Emission Map", 2D) = "white" {}

        [Enum(UnityEngine.Rendering.CullMode)] _Cull ("Cull", Float) = 2
        [Enum(UnityEngine.Rendering.CompareFunction)] _ZTest ("Z Test", Float) = 2

        _DitherTex ("Shadow DitherTex", 2D) = "white" {}
    }
    SubShader
    {
        Tags 
        { 
            "RenderType"="Opaque" 

            //"Queue" = "Geometry"
            "Queue" = "Transparent"

        }
        LOD 100
        ZTest LEqual
        ZWrite [_ZTest]
        Cull [_Cull]
        Blend SrcAlpha OneMinusSrcAlpha


        Pass
        {
            HLSLPROGRAM
            #pragma shader_feature _ROUGHTYPE_SMOOTHNESS_MAP _ROUGHTYPE_ROUGHNESS_MAP
            #pragma multi_compile _SURFACETYPE_SPECULAR _SURFACETYPE_METALLIC
            #if _SURFACETYPE_SPECULAR
                #define _SPECULAR
            #elif _SURFACETYPE_METALLIC
                #define _METALLIC
            #endif
            #pragma shader_feature TOGGLE_SM_COMBINE       
            #pragma multi_compile _MAP_NONE _MAP_USE
            #pragma shader_feature TOGGLE_EMISSION
            #pragma shader_feature TOGGLE_NORMAL

            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #pragma target 3.0
            #pragma multi_compile_instancing
            #pragma instancing_options renderinglayer

            #pragma shader_feature_local _NORMALTEX
            #pragma shader_feature_local_fragment _ProjectionAngleDiscardEnable
            #pragma shader_feature_local _UnityFogEnable
            #pragma shader_feature_local_fragment _FracUVEnable
            #pragma shader_feature_local_fragment _SupportOrthographicCamera

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _ADDITIONAL_LIGHT
            #pragma multi_compile _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile _ _ADDITIONAL_LIGHT_VERTEX
            #pragma multi_compile _ _ADDITIONAL_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _ADDITIONAL_LIGHT_CALCULATE_SHADOWS
            #pragma multi_compile _ _SHADOWS_SOFT
            #pragma multi_compile_fragment _ _LIGHT_COOKIES

            //Functions.hlsl
            #define _SOBEL _REMAP _GRAYSCALE

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"  
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/CommonMaterial.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/BSDF.hlsl"    
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Packing.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/GlobalIllumination.hlsl"
            //#include "./Shader/Functions.hlsl"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normalOS : NORMAL;
                float4 tangentOS    : TANGENT;
                float2 uv : TEXCOORD0;

#if LIGHTMAP_ON
                float2 uvLightmap   : TEXCOORD1;
#endif
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                DECLARE_LIGHTMAP_OR_SH(lightmapUV, vertexSH, 1);
                float4 vertex : SV_POSITION; //PositonCS

#if LIGHTMAP_ON
                float2 uvLightmap               : TEXCOORD1;
#endif
                float3 positionWS               : TEXCOORD2;
                half3  normalWS                 : TEXCOORD3;
                half4 tangentWS                 : TEXCOORD4;
                half4 biTangentWS               : TEXCOORD5;
                half3 viewDir                   : TEXCOORD6;
                half4 screenPosition            : TEXCOORD7;
                half4 fogFactorAndVertexLight   : TEXCOORD9; // x: fogFactor, yzw: vertex light
            };



            CBUFFER_START(UnityPerMaterial)         
                TEXTURE2D(_MainTex);
                SAMPLER(sampler_MainTex);
                TEXTURE2D(_NormalTex);
                SAMPLER(sampler_NormalTex);
                #if _MAP_NONE
                    TEXTURE2D(_SmoothnessMap);
                    SAMPLER(sampler_SmoothnessMap);   
                    TEXTURE2D(_MetallicMap);
                    SAMPLER(sampler_MetallicMap);   
                    TEXTURE2D(_AOMAP);      
                    SAMPLER(sampler_AOMAP);
                #elif _MAP_USE
                    TEXTURE2D(_MaskMap);
                    SAMPLER(sampler_MaskMap);
                #endif

                #if TOGGLE_EMISSION
                    TEXTURE2D(_EmissionMap);
                    SAMPLER(sampler_EmissionMap);               
                #endif
                TEXTURE2D(_CameraDepthTexture);
                SAMPLER(sampler_CameraDepthTexture);

                half4 _MainTex_ST;
                half4 _NormalTex_ST;
                half _NormalPower;
                half _ShadowAdjust;
                half _Smoothness;
                half _Metallic, _MetallicMapIntensity;
                half _AOIntensity;
                TEXTURE2D(_DitherTex);
                SAMPLER(sampler_DitherTex);
            CBUFFER_END

            v2f vert (appdata v)
            {
                v2f o;
                VertexPositionInputs vertexInput = GetVertexPositionInputs(v.vertex.xyz);
                VertexNormalInputs vertexNormalInput = GetVertexNormalInputs(v.normalOS, v.tangentOS);
                o.vertex = TransformObjectToHClip(v.vertex.xyz);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.positionWS = vertexInput.positionWS;
                o.normalWS = NormalizeNormalPerVertex(vertexNormalInput.normalWS);

                //NormalMap
                o.tangentWS.xyz = TransformObjectToWorldDir(v.tangentOS.xyz);
                float tangentSign = v.tangentOS.w * unity_WorldTransformParams.w;
                o.biTangentWS.xyz = cross(o.normalWS.xyz, o.tangentWS.xyz) * tangentSign;

                /* viewDir 메모
                * Perspective : return normalize(_WorldSpaceCameraPos.xyz - positionWS);
                * Othro : return -(-UNITY_MATRIX_V[2].xyz);      */              
                o.viewDir = normalize(GetWorldSpaceViewDir(o.positionWS));

                /*  float4 o = positionCS * 0.5f;
                    o.xy = float2(o.x, o.y * _ProjectionParams.x) + o.w;
                    o.zw = positionCS.zw;      screenPosition = o;  */
                o.screenPosition = ComputeScreenPos(o.vertex);

                OUTPUT_LIGHTMAP_UV(v.uvLightmap, unity_LightmapST, o.uvLightmap);
                OUTPUT_SH(o.normalWS.xyz, o.vertexSH);

                o.fogFactorAndVertexLight.x = ComputeFogFactor(o.vertex.z);
                o.fogFactorAndVertexLight.yzw =  VertexLighting(vertexInput.positionWS, vertexNormalInput.normalWS);
                return o;
            }


            inline float3 SafeNormalize1(float3 inVec)
            {
                return inVec * rsqrt(max(0.001f, dot(inVec, inVec)));
            }

            inline half DisneyDiffuse(half NdotV, half NdotL, half LdotH, half perceptualRoughness)
            {
                half fd90 = .5 + 2 * LdotH * LdotH * perceptualRoughness;
                half lightScatter = 1 + ((fd90 - 1) * pow(1-NdotL, 5));
                half viewScatter =  1 + ((fd90 - 1) * pow(1-NdotV, 5));
                return lightScatter * viewScatter;
            }
            // Oren-Nayar diffuse model
            half OrenNayarDiffuse(half NdotV,  half NdotL, half NdotH, half roughness)
            {
                half sigma2 = roughness * roughness;
                half A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
                half B = 0.45 * (sigma2 / (sigma2 + 0.09));

                half theta_r = acos(NdotV);
                half theta_i = acos(NdotL);

                half alpha = max(theta_r, theta_i);
                half beta = min(theta_r, theta_i);

                return A + B * NdotH * sin(alpha) * tan(beta);
            }

            half OrenNayarDiffuse2(half NdotV,  half NdotL, half LdotV, half roughness)
            {
                half s = LdotV - NdotL * NdotV;
                half t = lerp(1.0, max(NdotL, NdotV), step(0.0, s));
                half sigma2 = roughness * roughness;
                half A = 1.0 + sigma2 * (1.0 / (sigma2 + .13) + .5 / (sigma2 + .33));
                half B = .45 * sigma2 / (sigma2 + .09);
                return A + B * s  / t; 
            }

            half SubsurfaceScatteringDiffuse(half NdotL, half subsurfaceScattering, half thickness)
            {
                 half3 subsurfaceTerm =  exp(-subsurfaceScattering * thickness);
                 half3 diffuseTerm =  (1.0 / 3.1415927) * NdotL;
                 return diffuseTerm + subsurfaceTerm;
            }

            inline half GGXTerm(half NdotH, half roughness)
            {
                float a2 = roughness * roughness;   
                float d = (NdotH * a2 - NdotH) * NdotH + 1.0;

                return 3.1415926535 * a2  / (d *d +1e-7f);
            }

            inline half SmithJointGGXVisibilityTerm(half NdotL, half NdotV, half roughness)
            {
                half  a = roughness;
                half lambdaV = NdotL * (NdotV * (1 - a) + a);
                half lambdaL = NdotV * (NdotL * (1 - a) + a);
                return 0.5f / (lambdaL + lambdaV + 1e-4f);
            }



            //Fresnel Funtions          추가 및 변경 할 이유가 없음
            half3 Fresnel_Schlick(half3 F0, half cosTheta) //대표적  FresnelTerm, F0 = specColor
            {
	            return F0 + (1 - F0) * pow(1 - cosTheta, 5);
            }
            half3 Fresnel_Lerp(half3 F0, half3 F90, half cosA) //F0 = specColor, F90 = grazingTerm, cosA =  NdotV
            {
	            half t = pow( (1 - cosA), 5);   // ala Schlick interpoliation
	            return lerp (F0, F90, t);
            }

            float SchlickIORFresnelFunction(float ior,float LdotH)
            {
                float f0 = pow((ior-1)/(ior+1),2);
                return f0 +  (1 - f0) * pow(LdotH, 5);  
            }


            half SpecularHighlight(half NdotH, half LdotH, half normalizationTerm, half roughness2)
            {
                // BRDFspec = (D * V * F) / 4.0
                // D = roughness^2 / ( NoH^2 * (roughness^2 - 1) + 1 )^2
                // V * F = 1.0 / ( LoH^2 * (roughness + 0.5) )
                //normalizationTerm = (roughness + 0.5) * 4.0 rewritten as roughness * 4.0 + 2.0 to a fit a MAD.
                half d = NdotH * NdotH * (roughness2 - 1) +  1.00001f;
                half LdotH2 = LdotH * LdotH;
                half specularTerm = roughness2 / ((d * d) * max(0.0001, LdotH2) * normalizationTerm);

                #if defined (SHADER_API_MOBILE) || defined (SHADER_API_SWITCH)
                    specularTerm = specularTerm - HALF_MIN;
                    specularTerm = clamp(specularTerm, 0.0, 100.0); // Prevent FP16 overflow on mobiles
                #endif
                return specularTerm * specularTerm;
            }

            inline half3 DecodeHDR (half4 data, half4 decodeInstructions)
            {
                // Take into account texture alpha if decodeInstructions.w is true(the alpha value affects the RGB channels)
                half alpha = decodeInstructions.w * (data.a - 1.0) + 1.0;

                // If Linear mode is not supported we can skip exponent part
                #if defined(UNITY_COLORSPACE_GAMMA)
                    return (decodeInstructions.x * alpha) * data.rgb;
                #else
                    #if defined(UNITY_USE_NATIVE_HDR)
                        return decodeInstructions.x * data.rgb; // Multiplier for future HDRI relative to absolute conversion.
                    #else
                        return (decodeInstructions.x * pow(alpha, decodeInstructions.y)) * data.rgb;
                    #endif
                #endif
            }

            void UnpackMaskMap(Texture2D mask, sampler sampler_mask , half2 uv, out half o, out half r , out half m, out half s)
            {
                half4 combined = SAMPLE_TEXTURE2D(mask, sampler_mask, uv);
                //half4 combined = SAMPLE_TEXTURE2D(_MaskMap, sampler_LinearRepeat, uv);
                o = combined.r;
                r = combined.g;
                m = combined.b;
                s = combined.a;
            }


            half4 frag(v2f i) : SV_Target
            {
                float3 output;

                float4 albedo = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv);

                half smoothness = _Smoothness;
                half roughnessMap = 1;
                half metallicMap = 1;
                half AO = 1;
                half shadowMap = 1;
                #if _MAP_NONE
                    roughnessMap = SAMPLE_TEXTURE2D(_SmoothnessMap, sampler_LinearRepeat, i.uv);
                    metallicMap = SAMPLE_TEXTURE2D(_MetallicMap, sampler_LinearRepeat, i.uv);
                    AO = lerp(1, SAMPLE_TEXTURE2D(_AOMAP, sampler_LinearRepeat, i.uv), _AOIntensity);
                #elif _MAP_USE
                    UnpackMaskMap(_MaskMap, sampler_LinearRepeat, i.uv, AO, roughnessMap, metallicMap, shadowMap);
                    AO = lerp(1, AO, _AOIntensity);
                #else
                    AO = max(1, 1 - _AOIntensity * _AOIntensity);
                #endif

                half metallic = saturate(metallicMap * _MetallicMapIntensity) * _Metallic;
   /*
                #if defined _ROUGHTYPE_ROUGHNESS_MAP
                    smoothness *= (1 - roughnessMap);
                #elif defined _ROUGHTYPE_SMOOTHNESS_MAP
                    smoothness *= roughnessMap;
                #endif
                */


                //탄젠트 계산

                //Invert 필요
                half3 normal = float3(1,1,1);  
                #if TOGGLE_NORMAL
                    float3x3 TBN = float3x3(i.tangentWS.xyz, i.biTangentWS.xyz, i.normalWS.xyz);
                    TBN = transpose(TBN);
                    half4 normalTex = SAMPLE_TEXTURE2D(_NormalTex, sampler_NormalTex, i.uv * _NormalTex_ST.xy + _NormalTex_ST.zw);
                    //half3 normal = UnpackNormalScale(normalTex, (_NormalPower));   //Normal을 완벽하게 끄려면 z = 1이 되어야 함
 
                    normal.xy = normalTex.ag * 2.0 - 1.0;
                    // 이거 때문에 NormalMap 자국 남지만 없으면 Specular가 깨짐
                    normal.z = max(1.0e-16, sqrt(1.0 - saturate(dot(normal.xy, normal.xy))));     
                    normal.xy *= _NormalPower;
                    //normal = UnpackNormalScale(normalTex, (_NormalPower)); 
                    normal = mul(TBN, normal);
                #else
                    normal = i.normalWS;
                #endif



                /* ShadowNoise
                float2 shadowNoiseUV = i.uv * _ShadowNoiseMap_ST.xy + _ShadowNoiseMap_ST.zw;
                shadowNoiseUV.xy += _Time.y * 0.1;
                float4 shadowNoise = SAMPLE_TEXTURE2D(_ShadowNoiseMap, sampler_ShadowNoiseMap, shadowNoiseUV);
                float4 shadowCoord = TransformWorldToShadowCoord(i.positionWS + shadowNoise);
                */

                float4 shadowCoord;
#if defined(_MAIN_LIGHT_SHADOWS_SCREEN) && !defined(_SURFACE_TYPE_TRANSPARENT)
                shadowCoord =  ComputeScreenPos(i.vertex);
#else
                shadowCoord =  TransformWorldToShadowCoord(i.positionWS);
#endif

                Light mainLight = GetMainLight(shadowCoord);

                ShadowSamplingData shadowSamplingData = GetMainLightShadowSamplingData();	//! 쉐도우 감쇠값
                half4 shadowParams = GetMainLightShadowParams();
                float shadowAtten = SampleShadowmap(TEXTURE2D_ARGS(_MainLightShadowmapTexture, sampler_MainLightShadowmapTexture), shadowCoord, shadowSamplingData, shadowParams, false) ;


                float3 Cookie = float3(0, 0, 0);
#if defined(_LIGHT_COOKIES)
                Cookie = SampleMainLightCookie(shadowCoord);
                #if _TYPE_LIT
                    //shadowAtten = min(Cookie.x, shadowAtten );
                    shadowAtten = min( saturate(shadowAtten  + _ShadowAdjust), Cookie.x);
                #elif _TYPE_TOON
                    shadowAtten = saturate(min(Cookie.x, shadowAtten) + _ShadowAdjust); //그림자 보정
                #endif
#else
                shadowAtten = saturate(shadowAtten + _ShadowAdjust);
#endif

                //노말맵 적용
                half fNDotL = dot(mainLight.direction, normal);
                #if _TYPE_TOON
                    float3 lightCol = mainLight.color /* fNDotL*/ * mainLight.distanceAttenuation * (shadowAtten);   //MainLight
                #endif

                //BDSF                     


                /*  Based On Torrance-Sparrow micro-facet
                * BRDF = kD / pi + kS * (D * V * F) / 4
                * I = BRDF * NdotL
                *   
                * Normal Distribution Function
                * a) Blin-Phong
                * b) GGX
                * Geometric Shadow Function
                * a) SmithJointGGXVisibilityTerm
                * b) SmithBeckmanVisibilityTerm
                * Schick Approximation Fresnel 
                *   Fresnel_Schlick
                *   Fresnel_Lerp
                */

                half3 lightReflectDirection = reflect( -mainLight.direction, normal );
                half3 reflectVector = SafeNormalize1(reflect(-i.viewDir, normal)); //viewReflectDirection 
                half3 halfvector = SafeNormalize1(mainLight.direction.xyz + i.viewDir); //HalfVector or HalfDirection
                
                half  NdotL = dot(mainLight.direction, normal);                 half halfLambert = NdotL * .5 + .5;
                half NdotH = dot(normal, halfvector);
                half RdotV = dot(reflectVector, i.viewDir);
                half LdotH =  dot(mainLight.direction, halfvector); 
                half HdotV = dot(halfvector, i.viewDir);
                half NdotV = abs(dot(normal, i.viewDir));


                half perceptualRoughness = 1 - (_Smoothness);
                half roughness = max(0.002, perceptualRoughness * perceptualRoughness);



                
                #define kDieletricSpec half4(0.04, 0.04, 0.04, 1.0 - 0.04)

                //OneMinusReflectivity
                half oneMinusReflectivity  = kDieletricSpec.a - (metallic* kDieletricSpec.a);
                //Diffuse And Specular From Metallic
                float3 diffuseColor = albedo.rgb * (oneMinusReflectivity);
                float3 specColor = lerp(kDieletricSpec.xyz, albedo.xyz, metallic);


                half diffuseTerm = DisneyDiffuse(saturate(NdotV), saturate(NdotL), saturate(LdotH), roughness) * saturate(min(NdotL, shadowAtten));
                //half t = diffuseTerm;
                //diffuseTerm = OrenNayarDiffuse(saturate(NdotV), saturate(NdotL), saturate(NdotH), roughness) * saturate(min(NdotL, shadowAtten));
                //half t2 = diffuseTerm;
                //diffuseTerm = OrenNayarDiffuse2(saturate(NdotV), saturate(NdotL), saturate(dot(mainLight.direction, i.viewDir)), roughness) * saturate(min(NdotL, shadowAtten)) * mainLight.color;
                
                //half t3 = SubsurfaceScatteringDiffuse(saturate(NdotL), 1, .1);

                half3 ambient = SampleSH(normal); //문제있음


                half V = SmithJointGGXVisibilityTerm(saturate(NdotL), saturate(NdotV), roughness);      //Geometric
                half D = GGXTerm(saturate(NdotH), roughness);       //Distribution
                
                half normalizationTerm = (roughness + 0.5) * 4.0;
                //normalizationTerm = max(0.00001, (4 * (saturate(NdotL) * saturate(NdotV))));
                half3 specTerm = V * D; //  * 3.1415926535;     

                specTerm += SpecularHighlight(saturate(NdotH), saturate(LdotH), normalizationTerm, roughness);

                #ifdef UNITY_COLORSPACE_GAMMA
                    specTerm = sqrt(max(0.0001, specTerm));
                #endif


                specTerm = max(0 , specTerm * saturate(min(NdotL, shadowAtten)));
                specTerm *= any(specColor)? 1.0 : 0;
                 
                half3 aDiffuseTerm =half3(0,0,0);
                half3 aSpecTerm = half3(0,0,0);
                uint lightsCount = GetAdditionalLightsCount();

                for (int j = 0; j < lightsCount; j++)
                {
                    ShadowSamplingData aShadowSamplingData = GetAdditionalLightShadowSamplingData(j);
                    float shadowStrength = GetAdditionalLightShadowStrenth(j);
                    half s = AdditionalLightRealtimeShadow(j, i.positionWS);
                    Light addlitionalLight = GetAdditionalLight(j, i.positionWS, shadowCoord);  //이래야 작동함
                    half3 aHalfvector =  SafeNormalize1(normalize(addlitionalLight.direction) + i.viewDir); 
                    half aNdotL = saturate(dot(normal, normalize(addlitionalLight.direction)));
                    half aNdotV = saturate(NdotV);
                    half aLdotH = saturate(dot( normalize(addlitionalLight.direction), aHalfvector));
                    half addlitionalShadowAtten = SampleShadowmap(TEXTURE2D_ARGS(_AdditionalLightsShadowmapTexture, sampler_AdditionalLightsShadowmapTexture), shadowCoord, aShadowSamplingData, shadowStrength, false);

                    half3 aSpec = SpecularHighlight(saturate(dot(normal, aHalfvector)), aLdotH, normalizationTerm, .5);
                    
                    half adiffuseTerm = DisneyDiffuse(aNdotV, aNdotL, aLdotH,  .5);
                    //재사용
                    aHalfvector = saturate(min(aNdotL, addlitionalLight.shadowAttenuation)) *  saturate( pow(addlitionalLight.distanceAttenuation, 2) ) * addlitionalLight.color;
                    //aHalfvector = saturate(min(aNdotL, addlitionalLight.shadowAttenuation)) *  (addlitionalLight.distanceAttenuation) * addlitionalLight.color;
                    //aSpecTerm += aSpec * aHalfvector;
                    aDiffuseTerm += adiffuseTerm * aHalfvector;

                    //return addlitionalLight.shadowAttenuation > 0 ? 1 : 0;

                    //return addlitionalLight.shadowAttenuation;// * addlitionalLight.distanceAttenuation;
                    //return float4(saturate( pow(addlitionalLight.distanceAttenuation, 3) * .15) * addlitionalLight.color, 1);
                    //return addlitionalLight.shadowAttenuation * pow(addlitionalLight.distanceAttenuation, 3) ;
                }
                

                //float base = 1 - saturate(HdotV);
                //float exponential = pow(base, 5);
                //float frenel = exponential + specColor * (1 - exponential); 

                //Fresnel Term          Direct
                half3 fresnelTerm = Fresnel_Schlick(specColor, saturate(NdotV));   
                fresnelTerm = Fresnel_Schlick(specColor, saturate(LdotH));   
                //Fresnel Lerp          Indirect
                half grazingTerm = saturate(_Smoothness + (1-oneMinusReflectivity));        //문제없음
                half3 fresnelLerp = Fresnel_Lerp(specColor , grazingTerm, saturate(NdotV));



                half3 KD_IndirectLight = float3(1, 1, 1) - fresnelLerp;
                // return KD_IndirectLight.xyzz;
                KD_IndirectLight *= 1 - metallic;

                half3 diffuseIndirect = ambient * KD_IndirectLight;

                //Diffuse
                half3 diffuse = diffuseColor * (diffuseIndirect +  diffuseTerm * mainLight.color) + diffuseColor *  aDiffuseTerm * KD_IndirectLight;     


                half3 specular = specTerm * mainLight.color * fresnelTerm / normalizationTerm + aSpecTerm * fresnelLerp ;

                //return float4(specular, 1);

                //GlossyEnvironmentReflection
                //UnityGI gi =  GetUnityGI(_LightColor0.rgb, -mainLight.direction, worldNormal, i.viewDir, reflectVector, shadowAtten, 1- _Smoothness, i.positionWS.xyz);
                //PerceptualRoughnessToMipmapLevel = roughness * (1.7  - .7  * roughness) * UNITY_SPECCUBE_LOD_STEPS
                float4 envSample = SAMPLE_TEXTURECUBE_LOD(unity_SpecCube0,samplerunity_SpecCube0, reflectVector, PerceptualRoughnessToMipmapLevel(1 - _Smoothness));
		        float3 indirectSpecular = DecodeHDR(envSample, unity_SpecCube0_HDR);
                #if defined(UNITY_COLORSPACE_GAMMA)
                    #define unity_ColorSpaceDouble float4(2.0, 2.0, 2.0, 2.0)
                #else
                    #define unity_ColorSpaceDouble float4(4.59479380, 4.59479380, 4.59479380, 2.0)
                #endif
                //Reflection 안쓰면
                //indirectSpecular = _GlossyEnvironmentColor.rgb;


                //(Int)D(NdotH) * NdotH * Id(NdotL > 0) dH =  1 / (roughness^2 + 1)
                #ifdef UNITY_COLORSPACE_GAMMA
                    half surfaceReduction = 1 - 0.28 * roughness * perceptualRoughness;
                #else
                    half surfaceReduction = 1.0 / (roughness * roughness + 1);
                #endif
                
                half3 indirect = surfaceReduction * indirectSpecular * fresnelLerp * AO;
 
                output = float4(diffuse + specular + indirect, 1);

                //Backlight
                //col.xyz += EdgeHighlight(_CameraDepthTexture, sampler_CameraDepthTexture,i.uv, 0.01);


                return float4((output), albedo.w);
            }
            ENDHLSL
        }

        Pass        //Shadow Caster
        {
            Name "ShadowCaster"
            Tags
            {
                "LightMode" = "ShadowCaster"
            }
            HLSLPROGRAM
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
                        
            #pragma vertex ShadowPassVertex
            #pragma fragment ShadowPassFragment

            CBUFFER_START(UnityPerMaterial)
                TEXTURE2D(_MainTex);
                SAMPLER(sampler_MainTex);
                half4 _MainTex_TexelSize;
                float3 _LightDirection;
                float3 _LightPosition;
                TEXTURE2D(_DitherTex);
                SAMPLER(sampler_DitherTex);
                half4 _DitherTex_TexelSize;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float3 normalOS     : NORMAL;
                float2 texcoord     : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float2 uv       : TEXCOORD0;
                float4 positionCS   : SV_POSITION;
                half4 screenPosition : TEXCOORD1;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };


            float4 GetShadowPositionHClip(Attributes input) 
            {
                float3 os = input.positionOS.xyz;
                float3 positionWS = TransformObjectToWorld(os) ;
                //float3 positionWS = TransformObjectToWorld(input.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(input.normalOS);

            #if _CASTING_PUNCTUAL_LIGHT_SHADOW
                float3 lightDirectionWS = normalize(_LightPosition - positionWS);
            #else
                float3 lightDirectionWS = _LightDirection;
            #endif

                float4 positionCS = TransformWorldToHClip(ApplyShadowBias(positionWS, normalWS, lightDirectionWS))  ;

            #if UNITY_REVERSED_Z
                positionCS.z = min(positionCS.z, UNITY_NEAR_CLIP_VALUE);
            #else
                positionCS.z = max(positionCS.z, UNITY_NEAR_CLIP_VALUE);
            #endif
                return positionCS;
            }

            Varyings ShadowPassVertex(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);

                output.uv = input.texcoord;

                output.positionCS = GetShadowPositionHClip(input);
                output.screenPosition = ComputeScreenPos(output.positionCS);
                return output;
            }

            float Dithering(float2 uv)
            {
                //float2 uv = ScreenPosition.xy * _ScreenParams.xy;
                float DITHER_THRESHOLDS[16] =
                {
                    1.0 / 17.0,  9.0 / 17.0,  3.0 / 17.0, 11.0 / 17.0,
                    13.0 / 17.0,  5.0 / 17.0, 15.0 / 17.0,  7.0 / 17.0,
                    4.0 / 17.0, 12.0 / 17.0,  2.0 / 17.0, 10.0 / 17.0,
                    16.0 / 17.0,  8.0 / 17.0, 14.0 / 17.0,  6.0 / 17.0
                };
                uint index = (uint(uv.x) % 4) * 4 + uint(uv.y) % 4;
                return DITHER_THRESHOLDS[index];
            }

            half Dithering2(half2 uv)
            {
                // Bayer matrix (4x4)
                float bayerMatrix[16] = {
                    0.0,  8.0,  2.0, 10.0,
                    12.0, 4.0, 14.0, 6.0,
                    3.0, 11.0, 1.0,  9.0,
                    15.0, 7.0, 13.0, 5.0
                };
                // 디더링 좌표 계산
                int2 pixelCoords = int2(floor(uv)) % 4; // 4x4 매트릭스 반복
                int bayerIndex = pixelCoords.y * 4 + pixelCoords.x;
                return bayerMatrix[bayerIndex] / 16.0;
            }


            half4 ShadowPassFragment(Varyings input) : SV_TARGET
            {
                UNITY_SETUP_INSTANCE_ID(input);

            #if defined(_ALPHATEST_ON)
                Alpha(SampleAlbedoAlpha(input.uv, TEXTURE2D_ARGS(_BaseMap, sampler_BaseMap)).a, _BaseColor, _Cutoff);
            #endif



            float4 albedo = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, input.uv);

            //half2 renderScale = _ScreenSize.xy / _ScreenParams.xy;

            /*
            half2 uv = input.positionCS.xy / (_ScaledScreenParams.xy);
            uv *= (_ScreenParams.xy  );           
            half dither = Dithering(uv);
            */

            half dither = SAMPLE_TEXTURE2D(_DitherTex, sampler_DitherTex,  half2(1 - albedo.w, albedo.w) ).w;

            //clip(dither - .5);
            clip(albedo.w - (dither));
            //clip(albedo.w - (pow(dither, .75)));

            #if defined(LOD_FADE_CROSSFADE)
                LODFadeCrossFade(input.positionCS);
            #endif

                return 0;
            }
            ENDHLSL
        }

        Pass        //DepthOnly
        {
            Name "DepthOnly"
            Tags
            {
                "LightMode" = "DepthOnly"
            }

            // -------------------------------------
            // Render State Commands
            ZWrite On
            ColorMask R
            Cull[_Cull]

            HLSLPROGRAM
            #pragma target 2.0
                    #pragma multi_compile_instancing
            #define UNITY_SUPPORT_INSTANCING
            // -------------------------------------
            // Shader Stages
            #pragma vertex DepthOnlyVertex
            #pragma fragment DepthOnlyFragment

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local _ALPHATEST_ON
            #pragma shader_feature_local_fragment _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A

            // -------------------------------------
            // Unity defined keywords
            #pragma multi_compile_fragment _ LOD_FADE_CROSSFADE


            //--------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing
            #include_with_pragmas "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DOTS.hlsl"

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            //CBuffer
            CBUFFER_START(UnityPerMaterial)
                half _GrayscaleStrength;

                half4 _MainTex_ST;
                half4 _NormalTex_ST;
                half _HeightScale;
                //UNITY_DEFINE_INSTANCED_PROP( float4, _Position)
                //UNITY_DEFINE_INSTANCED_PROP(half, _Radius)
                float4 _Position;
                half _Radius;
                half _Softness;
                half _ColorStrength;

                //half4 _EmissionColor;
                half _EmissionStrength;
            CBUFFER_END



            // -------------------------------------
            // Includes
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/DepthOnlyPass.hlsl"
            ENDHLSL
        }

        // This pass is used when drawing to a _CameraNormalsTexture texture
        Pass        //DepthNormals
        {
            Name "DepthNormals"
            Tags
            {
                "LightMode" = "DepthNormals"
            }

            // -------------------------------------
            // Render State Commands
            ZWrite On
            Cull[_Cull]

            HLSLPROGRAM
            #pragma target 2.0
                    #pragma multi_compile_instancing
            #define UNITY_SUPPORT_INSTANCING
            // -------------------------------------
            // Shader Stages
            #pragma vertex DepthNormalsVertex
            #pragma fragment DepthNormalsFragment

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local _NORMALMAP
            #pragma shader_feature_local _PARALLAXMAP
            #pragma shader_feature_local _ _DETAIL_MULX2 _DETAIL_SCALED
            #pragma shader_feature_local _ALPHATEST_ON
            #pragma shader_feature_local_fragment _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A

            // -------------------------------------
            // Unity defined keywords
            #pragma multi_compile_fragment _ LOD_FADE_CROSSFADE

            // -------------------------------------
            // Universal Pipeline keywords
            #include_with_pragmas "Packages/com.unity.render-pipelines.universal/ShaderLibrary/RenderingLayers.hlsl"

            //--------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing
            #include_with_pragmas "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DOTS.hlsl"


            // -------------------------------------
            // Includes
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitDepthNormalsPass.hlsl"
            ENDHLSL
        }

        // This pass it not used during regular rendering, only for lightmap baking.
        Pass        //Meta 
        {
            Name "Meta"
            Tags
            {
                "LightMode" = "Meta"
            }

            // -------------------------------------
            // Render State Commands
            Cull Off

            HLSLPROGRAM
            #pragma target 2.0
            #pragma multi_compile_instancing

            // -------------------------------------
            // Shader Stages
            #pragma vertex UniversalVertexMeta
            #pragma fragment UniversalFragmentMetaLit
                      #define UNITY_SUPPORT_INSTANCING
            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local_fragment _SPECULAR_SETUP
            #pragma shader_feature_local_fragment _EMISSION
            #pragma shader_feature_local_fragment _METALLICSPECGLOSSMAP
            #pragma shader_feature_local_fragment _ALPHATEST_ON
            #pragma shader_feature_local_fragment _ _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A
            #pragma shader_feature_local _ _DETAIL_MULX2 _DETAIL_SCALED
            #pragma shader_feature_local_fragment _SPECGLOSSMAP
            #pragma shader_feature EDITOR_VISUALIZATION

            // -------------------------------------
            // Includes
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitMetaPass.hlsl"

            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
