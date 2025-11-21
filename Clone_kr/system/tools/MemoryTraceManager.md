# MemoryTraceManager 도구

## 목적
LLMunix 작업 실행 세션 중 에이전트 통신의 휘발성 메모리 추적을 관리합니다. 학습 및 패턴 인식을 위해 에이전트 상호작용을 캡처, 분석, 통합합니다.

## 도구 명세

```yaml
tool_name: "MemoryTraceManager"
category: "memory_management"
mode: ["EXECUTION", "SIMULATION"]
description: "에이전트 통신 추적 및 메모리 통합 추적 및 관리"
```

## 핵심 기능

### 1. 추적 기록
**기능**: `record_agent_communication`
**목적**: 작업 실행 중 에이전트 간 통신 캡처

**매개변수**:
- `session_id`: 현재 실행 세션의 고유 식별자
- `from_agent`: 소스 에이전트 이름/유형
- `to_agent`: 대상 에이전트 이름/유형
- `message_type`: ["request", "response", "notification", "error", "delegation"]
- `message_content`: 실제 통신 내용
- `context`: 현재 작업 컨텍스트 및 상태
- `timestamp`: 통신의 ISO 타임스탬프
- `execution_step`: 전체 작업 실행의 현재 단계

**도구 호출 형식**:
```
TOOL_CALL: MemoryTraceManager.record_agent_communication
PARAMETERS:
  session_id: "sess_20240321_143022"
  from_agent: "SystemAgent"
  to_agent: "VisionaryAgent"
  message_type: "request"
  message_content: "심장 모니터링 시스템을 위한 비전을 생성해주세요"
  context: "프로젝트 Aorta 초기화 단계"
  timestamp: "2024-03-21T14:30:22Z"
  execution_step: 2
```

### 2. 세션 관리
**기능**: `create_session`
**목적**: 새 메모리 추적 세션 초기화

**매개변수**:
- `project_name`: 프로젝트 이름 (예: "Project_aorta")
- `goal`: 실행 중인 고수준 목표
- `agent_list`: 세션에 참여하는 에이전트 목록

**기능**: `close_session`
**목적**: 세션 마무리 및 통합 분석 트리거

### 3. 메모리 통합
**기능**: `analyze_session_for_learning`
**목적**: 완료된 세션 추적을 분석하여 학습 추출

**매개변수**:
- `session_id`: 분석할 세션
- `consolidation_criteria`: 찾을 패턴

**추출 대상**:
- 새로운 에이전트 상호작용 패턴
- 성공적인 협업 전략
- 통신 병목 현상
- 발견된 지식 격차
- 창발적 문제 해결 접근 방식

## 메모리 저장 구조

### 단기 메모리 (휘발성)
**위치**: `projects/{project_name}/workspace/memory/traces/`

**파일 구조**:
```
traces/
├── session_[timestamp]/
│   ├── session_metadata.md      # 마크다운으로 된 세션 정보
│   ├── communication_log.jsonl  # 스트리밍을 위한 줄 구분 JSON
│   ├── agent_states.md          # 마크다운으로 된 에이전트 상태 스냅샷
│   ├── context_evolution.md     # 세션 중 컨텍스트 변화
│   └── execution_flow.md        # 마크다운으로 된 단계별 실행 추적
```

**세션 메타데이터 형식** (session_metadata.md):
```markdown
# 세션 메타데이터

**세션 ID**: sess_20240321_143022
**프로젝트**: Project_aorta
**목표**: 양자 심장 모니터링 비전 생성
**시작 시간**: 2024-03-21T14:30:22Z
**상태**: active

## 참여 에이전트
- SystemAgent (orchestrator)
- VisionaryAgent (vision_creator)
- MathematicianAgent (framework_developer)
- QuantumEngineerAgent (implementer)

## 구성
**모드**: EXECUTION
**우선순위**: high
**제약**: 표준 계산 제한
```

**실행 흐름 형식** (execution_flow.md):
```markdown
# 실행 흐름

## 단계 1: 초기화
**시간**: 2024-03-21T14:30:22Z
**에이전트**: SystemAgent
**작업**: 세션 초기화 및 프로젝트 컨텍스트 로드
**결과**: ✅ 성공

## 단계 2: 비전 요청
**시간**: 2024-03-21T14:30:25Z
**에이전트**: SystemAgent → VisionaryAgent
**작업**: 비전 생성 요청
**상태**: ⏳ 진행 중
```

**통신 로그 항목 형식** (communication_log.jsonl):
```json
{
  "timestamp": "2024-03-21T14:30:22Z",
  "step": 2,
  "from_agent": "SystemAgent",
  "to_agent": "VisionaryAgent",
  "message_type": "request",
  "content": "심장 모니터링 시스템을 위한 비전을 생성해주세요",
  "context_snapshot": "프로젝트 초기화 단계",
  "response_expected": true,
  "priority": "high"
}
```

### 장기 메모리 (지속적)
**위치**: `projects/{project_name}/memory/`

**파일 구조**:
```
memory/
├── learned_patterns.md          # 통합된 상호작용 패턴
├── agent_collaboration_map.md   # 효과적인 에이전트 조합
├── knowledge_discoveries.md     # 세션에서 얻은 새 인사이트
├── communication_templates.md   # 성공적인 통신 패턴
├── session_template.md          # 세션 기록 템플릿
└── session_summaries/           # 정제된 세션 학습
    ├── 2024-03-21_session_analysis.md
    └── ...
```

## 기존 메모리 시스템과의 통합

### 시스템 메모리 로그와의 연결
- 통합 결과가 `system/memory_log.md`에 제공됨
- 에이전트 통신 인사이트가 구조화된 경험 항목이 됨
- 교차 프로젝트 패턴이 식별되어 시스템 수준에서 저장됨

### QueryMemoryTool 통합
- MemoryTraceManager가 QueryMemoryTool에 통신 패턴 데이터 제공
- 과거 에이전트 상호작용 성공률이 향후 에이전트 선택에 정보 제공
- 통신 템플릿이 최적의 메시지 형식 제안

## 사용 예시

### 작업 실행 중
```
# SystemAgent가 VisionaryAgent에게 위임
TOOL_CALL: MemoryTraceManager.record_agent_communication
PARAMETERS:
  session_id: "aorta_vision_generation_001"
  from_agent: "SystemAgent"
  to_agent: "VisionaryAgent"
  message_type: "request"
  message_content: "심장 양자 모니터링을 위한 종합 비전 생성"
  context: "초기 프로젝트 범위 지정 단계"
  execution_step: 1

# VisionaryAgent가 비전으로 응답
TOOL_CALL: MemoryTraceManager.record_agent_communication
PARAMETERS:
  session_id: "aorta_vision_generation_001"
  from_agent: "VisionaryAgent"
  to_agent: "SystemAgent"
  message_type: "response"
  message_content: "[생성된 비전 문서 내용]"
  context: "비전 생성 완료"
  execution_step: 1
```

### 세션 통합
```
# 성공적인 세션 종료 시
TOOL_CALL: MemoryTraceManager.analyze_session_for_learning
PARAMETERS:
  session_id: "aorta_vision_generation_001"
  consolidation_criteria: ["successful_handoffs", "knowledge_creation", "communication_efficiency"]
```

**통합 출력**: 다음 마크다운 파일 업데이트:
- `learned_patterns.md` - 발견된 새 패턴
- `agent_collaboration_map.md` - 업데이트된 협업 메트릭
- `communication_templates.md` - 정제된 메시지 템플릿
- `session_summaries/`에 새 세션 요약 생성

## 캡처되는 학습 패턴

### 에이전트 상호작용 패턴
- 다른 작업 유형에 가장 잘 작동하는 에이전트 조합
- 최적의 통신 타이밍 및 시퀀싱
- 효과적인 위임 전략
- 지식 이전 메커니즘

### 통신 효과성
- 왕복을 줄이는 메시지 형식
- 오해를 방지하는 컨텍스트 공유
- 오류 통신 및 복구 패턴
- 성공적인 협업 템플릿

### 지식 진화
- 에이전트 상호작용을 통해 이해가 어떻게 심화되는지
- 멀티 에이전트 처리에서 나타나는 발견 패턴
- 효과적인 지식 합성 접근 방식
- 에이전트 협업에서 나오는 창의적 인사이트

## 비용 및 성능

### 저장소 최적화
- 통합 후 휘발성 추적 자동 삭제
- 장기 저장소를 위한 압축
- 학습 가치에 기반한 선택적 보유

### 분석 효율성
- 실시간 패턴 감지를 위한 스트림 처리
- 심층 분석을 위한 배치 통합
- 여러 세션 추적의 병렬 처리

## 오류 처리

### 추적 손상 복구
- 중요 통신에 대한 중복 저장소
- 부분 추적에서 자동 복구
- 추적이 불완전할 때 그레이스풀 디그레이데이션

### 분석 실패
- 기본 패턴 추출로 폴백
- 복잡한 세션에 대한 수동 검토 트리거
- 단순화된 기준으로 점진적 재시도
