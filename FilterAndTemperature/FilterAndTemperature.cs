using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace Assets.Shader
{
    public class FilterAndTemperature : MonoBehaviour
    {
        public ClampedFloatParameter filter = new ClampedFloatParameter(1.0f, 0.0f, 2.0f);

        // Temperature 값 (켈빈 단위, 기본 6500K, 1000~10000K 범위)
        public ClampedFloatParameter temperature = new ClampedFloatParameter(6500.0f, 1000.0f, 10000.0f);

        [SerializeField] private Material mat;

        Color TemperatureToColor(float kelvin)
        {
            float temp = kelvin / 100.0f;
            float r, g, b;
            if (temp <= 66)
            {
                r = 255;
                g = Mathf.Clamp(99.4708025861f * Mathf.Log(temp) - 161.1195681661f, 0, 255);
                b = temp <= 19 ? 0 : Mathf.Clamp(138.5177312231f * Mathf.Log(temp - 10) - 305.0447927307f, 0, 255);
            }
            else
            {
                r = Mathf.Clamp(329.698727446f * Mathf.Pow(temp - 60, -0.1332047592f), 0, 255);
                g = Mathf.Clamp(288.1221695283f * Mathf.Pow(temp - 60, -0.0755148492f), 0, 255);
                b = 255;
            }
            return new Color(r / 255f, g / 255f, b / 255f);
        }

        // Update is called once per frame
        void Update()
        {
            if (mat == null)
                return;

            // 셰이더에 파라미터 전달
            mat.SetFloat("_Filter", filter.value);

            // 켈빈 온도를 RGB 색상으로 변환 (간단한 근사 함수 사용)
            Color tempColor = TemperatureToColor(temperature.value);
            mat.SetColor("_TemperatureColor", tempColor);
        }
    }
}