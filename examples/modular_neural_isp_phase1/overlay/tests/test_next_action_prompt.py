"""The Agent task must dispatch the controller action instead of closing a pause."""
import json

import pytest

from tm_research import campaign, research
from test_slow_decisions import plateau, save, slow


@pytest.mark.parametrize('action,required_text', [
    ('wait', 'collect existing work'),
    ('diagnose', 'campaign review-packet'),
    ('propose', 'campaign submit'),
    ('finalize', 'campaign finalize'),
    ('report_stop', 'stop search'),
    ('request_scope_change', 'new authorized campaign'),
    ('confirm', 'authorized confirmation plan'),
])
def test_current_route_has_matching_agent_instruction(tmp_path, action, required_text):
    current = plateau(tmp_path)
    if action == 'propose':
        current['trials'][-1]['result_status'] = 'improved'
        save(tmp_path, current)
    elif action == 'confirm':
        current['frozen'] = current['best']
        current['status'] = 'finalized'
        save(tmp_path, current)
    elif action != 'diagnose':
        research.record_research_decision(tmp_path, slow(current, action))

    task = campaign.next_proposal(tmp_path)
    assert task['next_action']['action'] == action
    assert task['proposal_allowed'] is (action == 'propose')
    assert f'Current action: {action}.' in task['prompt']
    assert required_text in task['prompt']
    assert 'proposal_allowed is only a submission gate' in task['prompt']
    assert 'Stop when proposal_allowed is false' not in task['prompt']
    if action != 'propose':
        assert 'then call campaign submit' not in task['prompt']


def test_plateau_requests_event_review_before_another_proposal(tmp_path):
    plateau(tmp_path)
    task = campaign.next_proposal(tmp_path)
    assert task['next_action']['review_required'] is True
    assert not task['proposal_allowed']
    assert 'campaign record-decision' in task['prompt']
    assert 'trigger_id' in task['prompt']
    assert 'pause does not finish the research' in task['prompt']


def test_recorded_diagnosis_keeps_prompt_paused_after_reload(tmp_path):
    current = plateau(tmp_path)
    research.record_research_decision(tmp_path, slow(current, 'diagnose'))
    assert json.loads((tmp_path / 'campaign.json').read_text())['pending_slow_decision']
    task = campaign.next_proposal(tmp_path)
    assert task['next_action']['review_required'] is False
    assert task['next_action']['action'] == 'diagnose'
    assert 'diagnose' in task['prompt']
    assert 'explicit matching continuation' in task['prompt']
    assert 'Stop when proposal_allowed is false' not in task['prompt']


def test_submit_guard_does_not_become_an_instruction_to_freeze(tmp_path):
    # A legacy non-ready state still has a propose route: its submit gate must
    # remain effective without recreating the old false=>finalize instruction.
    current = plateau(tmp_path)
    current.pop('control')
    current['status'] = 'running'
    save(tmp_path, current)
    task = campaign.next_proposal(tmp_path)
    assert task['next_action']['action'] == 'propose'
    assert task['proposal_allowed'] is False
    assert 'Submission is blocked' in task['prompt']
    assert 'then call campaign submit' not in task['prompt']
    assert 'Stop when proposal_allowed is false' not in task['prompt']
