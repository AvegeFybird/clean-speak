import re

import pytest

from agent.tools import agent_tools


# 确保测试环境是干净的
@pytest.fixture(autouse=True)
def reset_external_data_cache():
    agent_tools.external_data.clear()
    agent_tools.available_record_months.clear()
    agent_tools.available_user_ids.clear()
    yield
    agent_tools.external_data.clear()
    agent_tools.available_record_months.clear()
    agent_tools.available_user_ids.clear()


# 程序是否能正确地从系统的环境变量中读取到用户 ID
def test_get_user_id_returns_bound_value(monkeypatch):
    monkeypatch.setenv(agent_tools.USER_ID_ENV_NAME, "1001")
    
    result = agent_tools.get_user_id.invoke({})

    assert result == "1001"


# 如果环境变量丢了或者填错了，程序能不能给出正确的报错或提示信息
def test_get_user_id_reports_missing_or_invalid_value(monkeypatch):
    # 测试“环境变量缺失”
    monkeypatch.delenv(agent_tools.USER_ID_ENV_NAME, raising=False)
    missing_result = agent_tools.get_user_id.invoke({})

    # 测试“用户不存在”
    monkeypatch.setenv(agent_tools.USER_ID_ENV_NAME, "9999")
    invalid_result = agent_tools.get_user_id.invoke({})

    assert agent_tools.USER_ID_ENV_NAME in missing_result
    assert "不存在" in invalid_result


def test_get_user_location_returns_bound_city_or_missing_prompt(monkeypatch):
    # 测试“正常获取”
    monkeypatch.setenv(agent_tools.USER_CITY_ENV_NAME, "东莞")
    assert agent_tools.get_user_location.invoke({}) == "东莞"

    # 测试“环境变量缺失”
    monkeypatch.delenv(agent_tools.USER_CITY_ENV_NAME, raising=False)
    result = agent_tools.get_user_location.invoke({})
    assert agent_tools.USER_CITY_ENV_NAME in result


# 返回的时间格式符合标准
def test_get_current_month_format():
    result = agent_tools.get_current_month.invoke({})

    # 格式为 YYYY-MM
    assert re.fullmatch(r"\d{4}-\d{2}",result)


def test_fetch_external_data_returns_record_and_fallback_month():
    # 测试“精确匹配”
    exact_result = agent_tools.fetch_external_data.invoke(
        {"user_id": "1001", "month": "2025-01"}
    )
    fallback_result = agent_tools.fetch_external_data.invoke(
        {"user_id": "1001", "month": "2026-04"}
    )

    assert "覆盖率" in exact_result
    assert "2026-04" in fallback_result
    assert "已自动改查最近可用月份" in fallback_result

# 如果用户不存在，程序能不能给出正确的报错或提示信息
def test_fetch_external_data_reports_invalid_user():
    result = agent_tools.fetch_external_data.invoke(
        {"user_id": "9999", "month": "2025-01"}
    )

    assert "没有找到用户ID" in result


def test_csv_helpers_return_users_and_months():
    assert "1001" in agent_tools.get_avalible_user_ids()
    assert "2025-01" in agent_tools.get_avalible_record_months()
    assert "2025-12" in agent_tools.get_user_available_months("1001")
