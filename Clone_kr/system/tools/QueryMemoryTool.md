# 메모리 쿼리 도구

**구성요소 유형**: Tool
**버전**: v2
**상태**: [REAL] - 프로덕션 준비됨
**Claude 도구 매핑**: Read, Grep, Bash, Task

## 목적

QueryMemoryTool은 SystemAgent와 시스템의 MemoryAnalysisAgent 사이의 브릿지 역할을 하며, 작업 계획 및 실행 중 메모리 조회를 위한 표준화된 인터페이스를 제공합니다. 이는 SystemAgent가 개선된 의사결정과 적응형 동작을 위해 과거 경험을 활용할 수 있게 합니다.

## 핵심 기능

### 메모리 조회 인터페이스
- 메모리 쿼리를 위한 간단하고 표준화된 인터페이스 제공
- 쿼리 형식화 및 응답 파싱 처리
- SystemAgent 워크플로우에 원활하게 통합
- 구조화된 쿼리와 자연어 쿼리 모두 지원

### 쿼리 최적화
- 현재 작업 컨텍스트에 따라 관련 필터 자동 제안
- 더 빠른 응답 시간을 위해 쿼리 매개변수 최적화
- 계산 오버헤드를 줄이기 위해 빈번한 쿼리 캐시
- 빈 결과에 대한 쿼리 개선 제안 제공

### 컨텍스트 통합
- 메모리 쿼리에 현재 작업 컨텍스트 자동 포함
- 현재 제약을 과거 제약 패턴에 매핑
- 목표 유사성에 따라 관련 경험 식별
- 작업 특화 메모리 조회 제공

## 입력 명세

```yaml
query: string               # 과거 경험에 대한 자연어 질문
context:                   # 관련성을 위한 현재 작업 컨텍스트
  goal: string             # 현재 작업 목표
  task_type: string        # 작업 유형 (legal, research, creative 등)
  constraints: {}          # 현재 제약 설정
  components_considered: []# 현재 작업에 고려 중인 구성요소
filters:                   # 선택적 필터 (제공되지 않으면 자동 생성)
  outcome: string
  tags: []
  date_range: {}
  sentiment: string
  cost_range: {}
  error_threshold: number
options:                   # 쿼리 동작 옵션
  max_experiences: number  # 분석할 경험 수 제한
  include_failures: boolean# 실패한 경험 포함 여부
  priority_focus: string   # "recent", "successful", "similar", "all"
  confidence_threshold: number # 권장 사항의 최소 신뢰도
```

## 출력 명세

```yaml
query_id: string           # 이 쿼리의 고유 식별자
memory_analysis:           # MemoryAnalysisAgent의 응답
  analysis_summary: string
  relevant_experiences: []
  key_insights: []
  recommendations: []
  confidence_score: number
  behavioral_suggestions: {}
actionable_insights:       # 즉시 사용을 위해 처리됨
  constraint_adjustments: {}# 특정 제약 수정
  component_recommendations: []# 과거에 기반한 권장 구성요소
  risk_warnings: []        # 피해야 할 잠재적 문제
  success_patterns: []     # 복제할 패턴
query_metadata:           # 쿼리 실행에 대한 정보
  experiences_found: number
  processing_time: number
  cost_estimate: number
  cache_hit: boolean
recommendations:          # SystemAgent를 위한 다음 단계
  apply_constraints: {}    # 업데이트할 제약
  create_components: []    # 생성할 새 구성요소
  consultation_follow_ups: []# 고려할 추가 쿼리
```

## 실행 로직

### 단계 1: 쿼리 준비
1. **컨텍스트 분석**: 현재 작업 목표 및 제약 분석
2. **필터 생성**: 제공되지 않은 경우 관련 필터 자동 생성
3. **쿼리 최적화**: 성능 및 관련성을 위해 쿼리 최적화
4. **캐시 확인**: 유사한 쿼리가 최근에 캐시되었는지 확인

### 단계 2: 메모리 분석 실행
1. **쿼리 형식화**: 입력을 memory-analysis-agent 형식으로 변환
2. **분석 실행**: 형식화된 쿼리로 Task 도구를 사용하여 memory-analysis-agent 호출
3. **응답 파싱**: 분석에서 인사이트 및 권장 사항 추출
4. **결과 검증**: 응답 품질 및 관련성 보장

### 단계 3: 응답 처리
1. **인사이트 변환**: 인사이트를 실행 가능한 SystemAgent 안내로 변환
2. **제약 매핑**: 과거 패턴을 현재 제약 옵션에 매핑
3. **위험 평가**: 과거 실패에 기반한 잠재적 문제 식별
4. **성공 패턴 추출**: 복제 가능한 성공 패턴 식별

### 단계 4: 출력 생성
1. **응답 형식화**: SystemAgent 소비를 위한 출력 구조화
2. **권장 사항 생성**: 특정 작업 권장 사항 생성
3. **결과 캐시**: 잠재적 재사용을 위해 결과 저장
4. **쿼리 기록**: 향후 최적화를 위해 쿼리 기록

## Claude 도구 매핑

### 구현 패턴
```markdown
Action: Read system/memory_log.md
Observation: [컨텍스트 인식을 위한 메모리 로그 콘텐츠]

Action: memory-analysis-agent 호출을 위한 Task 도구
Parameters:
  description: "인사이트를 위한 메모리 쿼리"
  prompt: "[컨텍스트 및 필터가 있는 형식화된 쿼리]"
  subagent_type: "memory-analysis-agent"
Observation: [서브 에이전트의 구조화된 메모리 분석]

Action: [SystemAgent를 위한 결과 처리 및 형식화]
Observation: [사용 준비된 구조화되고 실행 가능한 인사이트]
```

### 도구 사용 전략
- **Read**: 메모리 로그 로드 및 관련 항목 추출
- **Task**: 특화된 memory-analysis-agent 서브 에이전트에게 메모리 분석 위임
- **Grep**: 특정 패턴 검색 및 경험 필터링
- **Bash**: 필요시 복잡한 분석을 위한 고급 텍스트 처리

## 예제 사용 시나리오

### 시나리오 1: 법적 분석 작업 계획
```yaml
# 입력
query: "법적 문서 분석에 어떻게 접근해야 하나요?"
context:
  goal: "공급업체 계약의 준수 위험 분석"
  task_type: "legal"
  constraints:
    user_sentiment: "neutral"
    priority: "quality"
```

```yaml
# 출력
actionable_insights:
  constraint_adjustments:
    error_tolerance: "strict"
    human_review_trigger_level: "low"
    active_persona: "detailed_analyst"
  component_recommendations:
    - "RiskDetectorTool"
    - "ComplianceAnalysisAgent"
  risk_warnings:
    - "도메인 특화 도구 누락으로 인한 이전 실패"
    - "표준 에이전트는 법적 패턴 인식이 부족함"
  success_patterns:
    - "분석 시작 전 특화 도구 생성"
    - "법적 작업에 상세 분석가 페르소나 사용"
```

### 시나리오 2: 감정 기반 적응
```yaml
# 입력
query: "사용자가 느린 진행에 불만스러워 보입니다. 어떻게 적응해야 하나요?"
context:
  goal: "마케팅 콘텐츠 생성"
  constraints:
    user_sentiment: "frustrated"
    priority: "comprehensiveness"
```

```yaml
# 출력
actionable_insights:
  constraint_adjustments:
    priority: "speed_and_clarity"
    active_persona: "concise_assistant"
    human_review_trigger_level: "low"
  success_patterns:
    - "불만이 감지되면 속도 중심 실행으로 전환"
    - "자주 진행 상황 업데이트 제공"
    - "중간 산출물 제공"
```

### 시나리오 3: 구성요소 선택 안내
```yaml
# 입력
query: "연구 작업에 가장 잘 작동하는 구성요소는 무엇인가요?"
context:
  goal: "AI 트렌드 연구 및 보고서 생성"
  task_type: "research"
```

```yaml
# 출력
component_recommendations:
  - "RealWebFetchTool" # 연구에 95% 성공률
  - "RealSummarizationAgent" # 높은 품질 분석 점수
  - "ResearchAnalysisAgent" # 트렌드 식별에 특화됨
success_patterns:
  - "병렬 페칭을 통한 다중 소스 연구"
  - "신뢰도 메트릭이 있는 구조화된 출력"
  - "종합 분석 후 요약 제공"
```

## SystemAgent와의 통합

QueryMemoryTool은 다음 SystemAgent 단계에서 사용됩니다:

### 계획 단계 통합
```markdown
## 계획 단계 (향상됨)
1. 사용자 목표 및 제약 파싱
2. **메모리 쿼리**: QueryMemoryTool을 사용하여 관련 과거 경험 찾기
3. **인사이트 적용**: 메모리 인사이트를 계획 생성에 통합
4. **제약 조정**: 권장 사항에 따라 constraints.md 업데이트
5. 과거 컨텍스트와 함께 실행 계획 생성
```

### 중간 실행 조회
```markdown
## 오류 복구 (향상됨)
1. 실행 문제 감지
2. **메모리 쿼리**: "유사한 오류가 이전에 어떻게 처리되었나요?"
3. **복구 적용**: 권장 복구 전략 사용
4. 복구 인사이트에 따라 제약 업데이트
```

## 성능 최적화

### 캐싱 전략
- 중복 분석을 피하기 위해 쿼리 결과를 1시간 동안 캐시
- 캐시 키에 쿼리 해시, 컨텍스트 지문, 메모리 로그 버전 포함
- 메모리 로그가 업데이트되면 캐시 무효화

### 쿼리 효율성
- 별도로 지정하지 않으면 가장 최근 100개 경험으로 분석 자동 제한
- 상세 분석 전 빠른 필터링을 위해 grep 사용
- 여러 인사이트가 필요할 때 유사한 쿼리 배치

### 비용 관리
- 실행 전 쿼리 비용 추정
- 비싼 쿼리에 대한 비용-편익 분석 제공
- 간단한 질문을 위한 가벼운 "빠른 조회" 모드 지원

## 오류 처리

### 메모리 접근 문제
- **빈 메모리 로그**: 기본 권장 사항과 함께 그레이스풀 디그레이데이션 제공
- **손상된 항목**: 손상된 데이터 건너뛰고 데이터 품질 문제 보고
- **접근 오류**: 메모리 사용 불가 시 기본 휴리스틱으로 폴백

### 쿼리 처리 오류
- **불명확한 쿼리**: 쿼리 개선 제안 제공
- **결과 없음**: 더 넓은 검색 기준 또는 관련 쿼리 제안
- **낮은 신뢰도**: 불확실한 권장 사항에 대한 인간 검증 요청

### 응답 검증
- **일관성 없는 권장 사항**: 검토를 위해 상충되는 인사이트 표시
- **오래된 패턴**: 최근 경험에 더 높은 가중치 부여
- **컨텍스트 불일치**: 과거 컨텍스트가 크게 다를 때 경고

## 향후 개선 사항

### 고급 쿼리 유형
- **예측 쿼리**: "이 접근 방식의 가능한 결과는 무엇인가요?"
- **비교 분석**: "접근 방식 A가 접근 방식 B와 역사적으로 어떻게 비교되나요?"
- **트렌드 분석**: "이 작업 유형에 대한 성능이 시간에 따라 어떻게 변했나요?"

### 지능형 자동화
- **사전 조회**: 명시적 요청 없이 메모리 자동 쿼리
- **실시간 적응**: 메모리 인사이트에 따라 지속적으로 제약 조정
- **학습 최적화**: 결과 피드백에 따라 쿼리 정확도 개선

### 향상된 통합
- **멀티 에이전트 조회**: 다른 에이전트를 대신하여 메모리 쿼리
- **교차 작업 학습**: 다른 작업 유형의 인사이트 적용
- **협업 메모리**: 여러 SystemAgent 인스턴스 간 인사이트 공유
