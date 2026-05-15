import os
import json
import csv
from datetime import datetime
from pathlib import Path


from langchain_core.tools import tool

from urllib.parse import urlencode #转换为查询字符串
from urllib.request import urlopen #url请求

from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.config_handler import get_abs_path


rag = None


external_data: dict[str, dict[str, dict[str, str]]] = {}
available_record_months: set[str] = set()
available_user_ids: set[str] = set()

runtime_state = {
    "last_rag_sources": [],
    "report_mode": False,
    "last_external_data": {},
}

USER_ID_ENV_NAME = "AGENT_USER_ID"
USER_CITY_ENV_NAME = "AGENT_USER_CITY"

WEATHER_CODE_MAP = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "有雾",
    48: "有霜雾",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "较强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "冰粒",
    80: "阵雨",
    81: "较强阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "伴有冰雹的雷暴",
    99: "强雷暴并伴有冰雹",
}

def reset_runtime_state() -> None:
    runtime_state["last_rag_sources"] = []
    runtime_state["report_mode"] = False
    runtime_state["last_external_data"] = {}

def mark_report_mode() -> None:
    runtime_state["report_mode"] = True

def get_runtime_state() ->dict[str, object]:
    return {
        "last_rag_sources" : list(runtime_state["last_rag_sources"]),
        "report_mode" : bool(runtime_state["report_mode"]),
        "last_external_data" : runtime_state["last_external_data"],
    }

def _read_env_value(env_name: str) -> str:
    return os.getenv(env_name, "").strip()

# RAG 懒加载
def _get_rag_service():
    global rag
    if rag is None:
        from rag.rag_service import RagSummarizeService

        rag = RagSummarizeService()
    return rag

@tool(description="从向量知识库中检索扫地机器人相关知识并生成总结。")
def rag_summarize(query: str)-> str:
    result = _get_rag_service().rag_summarize(query)
    runtime_state["last_rag_sources"] = result["sources"]
    return result["answer"]
    

#从网络上抓取 JSON 数据
def _fetch_json(url: str) ->dict:
    with urlopen(url,timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))
    
#获取城市经纬度
def _geocode_city(city: str) -> dict | None:
    query = urlencode({"name": city, "count" : 1, "language" : "zh", "format" : "json"})
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?{query}"
    geo_data = _fetch_json(geo_url)
    results = geo_data.get("results") or []
    return results[0] if results else None
     
#获取天气信息,输出转换器
def _format_weather(city_name: str,weather_data: dict) -> str:
    current = weather_data.get("current",{})
    daily = weather_data.get("daily",{})

    weather_code = current.get("weather_code")
    if weather_code is None:
        weather_code = current.get("weathercode")
    weather_desc = WEATHER_CODE_MAP.get(weather_code,f"天气代码{weather_code}")

    temperature = current.get("temperature_2m")
    wind_speed = current.get("wind_speed_10m")
    humidity = current.get("relative_humidity_2m")
    if humidity is None:
        humidity = current.get("relativehumidity_2m")

    # 获取降水概率
    precipitation_probability = None
    daily_precipitation = daily.get("precipitation_probability_max") or []
    if daily_precipitation:
        precipitation_probability = daily_precipitation[0]

    parts = [f"城市{city_name}当前{weather_desc}"]
    if temperature is not None:
        parts.append(f"气温{temperature}摄氏度")
    if wind_speed is not None:
        parts.append(f"风速{wind_speed}公里/小时")
    if humidity is not None:
        parts.append(f"相对湿度{humidity}%")
    if precipitation_probability is not None:
        parts.append(f"今日降水概率{precipitation_probability}%")

    return ",".join(parts)



@tool(description="基于Open-Meteo查询指定城市实时天气并返回中文结果")
def get_weather(city: str)-> str:
    city = city.strip()
    if not city:
        return "未提供城市名称,无法查询天气"
    
    try:
        location = _geocode_city(city)
        if not location:
            return f"未找到城市{city}，请尝试使用更完整的城市名称。"
        
        weather_query = urlencode(
            {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current" : "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily" : "precipitation_probability_max",
                "forecast_days": 1,
                "timezone": "auto",
            }
            
        )
        weather_url = f"https://api.open-meteo.com/v1/forecast?{weather_query}"
        weather_data = _fetch_json(weather_url)
        return _format_weather(location.get("name",city), weather_data)
    except Exception as e:
        logger.error(f"[get_weather] 查询城市 {city} 天气失败: {str(e)}")
        return f"暂时无法查询城市{city}的实时天气,请稍后尝试。"


@tool(description="获取当前绑定的城市名称。若环境变量未配置，则明确返回缺失提示，不允许编造城市。")
def get_user_location() -> str:
    city = _read_env_value(USER_CITY_ENV_NAME)
    if city:
        return city
    return f"当前未绑定城市信息，请先配置环境变量 {USER_CITY_ENV_NAME}。"


@tool(description="获取当前绑定的用户ID。若环境变量缺失或无效，则明确返回提示，不允许随机生成。")
def get_user_id()->str:
    generate_external_data()
    user_id = _read_env_value(USER_ID_ENV_NAME)
    if not user_id:
        return f"当前未绑定用户ID，请先配置环境变量 {USER_ID_ENV_NAME}。"
    if user_id not in available_user_ids:
        return f"当前绑定的用户ID“{user_id}”在使用记录中不存在，请检查环境变量 {USER_ID_ENV_NAME}。"
    return user_id
    


@tool(description="获取系统当前月份，返回 YYYY-MM 格式字符串。")
def get_current_month()->str:
    return datetime.now().strftime("%Y-%m")

def generate_external_data():
    if external_data:
        return
    
    external_data_path = get_abs_path(agent_conf["external_data_path"])

    if not os.path.exists(external_data_path):
        raise FileExistsError(f"外部数据目录{external_data_path}不存在")
    
    with open(external_data_path,"r",encoding="utf-8", newline= "") as file:
        reader = csv.DictReader(file)
        for row in reader:
            user_id = (row.get("用户ID") or "").strip()
            month = (row.get("时间") or "").strip()
            if not user_id or not month:
                continue

            available_user_ids.add(user_id)
            available_record_months.add(month)
            external_data.setdefault(user_id, {})
            external_data[user_id][month] = {
                "特征": (row.get("特征") or "").strip(),
                "效率": (row.get("清洁效率") or "").strip(),
                "耗材": (row.get("耗材") or "").strip(),
                "对比": (row.get("对比") or "").strip(),
            }
      
def get_avalible_record_months() -> list[str]:
    generate_external_data()
    return sorted(available_record_months)

def get_avalible_user_ids() -> list[str]:
    generate_external_data()
    return sorted(available_user_ids)

def get_user_available_months(user_id: str) -> list[str]:
    generate_external_data()
    return sorted(external_data.get(user_id,{}).keys())

# 把形如 YYYY-MM 的月份字符串解析成 (年, 月) 元组
def _parse_month_value(month: str) -> tuple[int, int] | None:
    try:
        year_text, month_text = month.split("-", maxsplit=1)
        year = int(year_text)
        month_num = int(month_text)
    except ValueError:
        return None
    
    if 1 <= month_num <= 12:
        return year, month_num
    return None

# 计算两个月份之间相差多少个月
def _month_distance(month_a: str, month_b: str) -> int | None:
    parsed_a = _parse_month_value(month_a)
    parsed_b = _parse_month_value(month_b)
    if not parsed_a or not parsed_b:
        return None

    year_a, month_num_a = parsed_a
    year_b, month_num_b = parsed_b
    return abs((year_a * 12 + month_num_a) - (year_b * 12 + month_num_b))


def resovle_nearest_available_month(month: str) -> str | None:
    months = get_avalible_record_months()
    if not months:
        return None
    if month in available_record_months:
        return month
    
    #一个空列表，用来保存候选月份及其距离
    ranked_months: list[tuple[int, str]] = []
    for candidate in months:
        distance = _month_distance(month, candidate)
        if distance is None:
            continue
        ranked_months.append((distance, candidate))

    if ranked_months:
        ranked_months.sort(key= lambda item: (item[0], item[1])) #距离优先、月份次优先
        return ranked_months[0][1]
    
    return months[-1]


def resolve_previous_user_month(user_id: str, month: str) -> str | None:
    user_months = get_user_available_months(user_id)
    current = _parse_month_value(month)
    if not current: #解析失败
        return None
    
    previous_months: list[tuple[int, str]] = []
    current_index = current[0] * 12 + current[1]
    for candidate in user_months:
        parsed = _parse_month_value(candidate)
        if not parsed: #解析失败
            continue
        candidate_index = parsed[0] * 12 + parsed[1]
        if candidate_index < current_index: #小于当前月份
            previous_months.append((candidate_index, candidate))

        if not previous_months: #没有小于当前月份的月份
            return None
        previous_months.sort(reverse=True)
        return previous_months[0][1]



def get_external_record(user_id: str, month: str) -> dict[str, str] | None:
    generate_external_data()
    return external_data.get(user_id,{}).get(month)

#环境检查和运行诊断
def get_runtime_banding_diagnostics() -> dict[str,object]:
    generate_external_data()

    user_id = _read_env_value(USER_ID_ENV_NAME)
    city = _read_env_value(USER_CITY_ENV_NAME)
    default_month = datetime.now().strftime("%Y-%m")
    months = get_avalible_record_months()

    missing: list[str] = []
    invalid: list[str] = []

    if not user_id:
        missing.append(f"环境变量`{USER_ID_ENV_NAME}` 未配置。")
    elif user_id not in available_user_ids:
        invalid.append(f"环境变量 `{USER_ID_ENV_NAME}` 当前值“{user_id}”未命中 `records.csv`。")

    if not city:
        missing.append(f"环境变量 `{USER_CITY_ENV_NAME}` 未配置。")

    if months and default_month not in available_record_months:
        fallback_month = resovle_nearest_available_month(default_month)
        invalid.append(
            f"系统当前月份 `{default_month}` 不在 `records.csv` 可查询月份内，将自动回退到最近可查月份 `{fallback_month}`。"
        )
    
    return {
        "binding": {
            "user_id": user_id,
            "city": city,
            "default_month": default_month,
        },
        "available_months": months,
        "missing": missing,
        "invalid": invalid,
    }





@tool(description="从外部系统中获取用户指定月份的使用记录；若月份不可查，会自动回退到最近可查月份并返回说明。")
def fetch_external_data(user_id: str,month: str)->str:
    """
    1.用户是否存在
    2.请求月份是否在全局月份范围内
    3.如果不在，能否回退到最近月份
    4.回退后的月份在该用户名下是否有记录
    5.成功时返回 JSON；若发生回退则附带说明
    """
    generate_external_data()

    if user_id not in external_data:
        logger.warning("[fetch_external_data] 用户 %s 不存在于使用记录中", user_id)
        return f"当前没有找到用户ID“{user_id}”的使用记录，请先确认绑定的 {USER_ID_ENV_NAME} 是否有效。"
    
    requested_month = month
    if month not in available_record_months:
        fallback_month = resovle_nearest_available_month(month)
        logger.warning(
            "[fetch_external_data] 月份 %s 不存在于 records.csv 中，自动回退到 %s",
            month,
            fallback_month,
        )
        if not fallback_month:
            return "当前没有可查询的使用记录月份，请先检查外部数据文件。"
        month = fallback_month

    user_months = get_user_available_months(user_id)
    if month not in external_data[user_id]:
        logger.warning("[fetch_external_data] 未查询到用户 %s 在 %s 的使用记录", user_id, month)
        return (
            f"用户“{user_id}”在“{month}”没有使用记录。"
            f"该用户可查询月份：{'、'.join(user_months)}。"
        )
    # 把找到的这条用户月份记录 (Python 对象) 转成 JSON 字符串
    payload = json.dumps(external_data[user_id][month], ensure_ascii=False)
    runtime_state["last_external_data"] = {
        "user_id": user_id,
        "requested_month": requested_month,
        "actual_month": month,
        "record": dict(external_data[user_id][month]),
    }
    if requested_month != month:
        return (
            f"原请求月份“{requested_month}”暂无记录，已自动改查最近可用月份“{month}”。"
            f"\n{payload}"
        )
    return payload


    
@tool(description="无入参,无返回值,调用后触发中间件自动为报告生成的场景动态注入上下文信息,为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"

def get_external_data_path() -> str:
    return str(Path(get_abs_path(agent_conf["external_data_path"])))

if __name__ == "__main__":
    print(fetch_external_data.invoke({"user_id":"1001","month": "2025-01"}))



