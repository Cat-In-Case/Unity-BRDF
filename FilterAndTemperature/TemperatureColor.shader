Shader "Hidden/Custom/SolarColor"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _TemperatureColor("Temperature Color", Color) = (1,1,1,1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            Name "SolarColor"
            ZTest Always 
            Cull Off 
            ZWrite On

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                output.positionCS = TransformObjectToHClip(input.vertex);
                output.uv = input.uv;
                return output;
            }

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);

            float _Filter;
            float4 _TemperatureColor; // RGB 값로 온도 보정

            float4 Frag(Varyings input) : SV_Target
            {
                float4 color = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, input.uv);
                // Filter를 곱해 색상의 강도를 조절
                color.rgb *= _Filter;
                // Temperature 보정을 위해 기존 색상과 temperature 색상을 보간 (여기서는 50% 혼합 예시)
                color.rgb = lerp(color.rgb, _TemperatureColor.rgb, 0.5);
                return color;
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/BlitCopy"
}
