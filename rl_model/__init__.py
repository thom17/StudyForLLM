"""
2024-07-17
Reinforcement learning
대략적으로 필요한 구조를 파악? 했으니 정리하자..

아래는 tuto에서 작성하다 필요하다 판단된 부분아다.
* RandChart <-> SimpleRandChartState <-> TradeChartAgent 의 데이터 교환 및 처리 루틴

일단 하나의 act의 기준이 되는 State는 별도의 클래스로 만들어 관리하는게 편할 것 같다.

*env_state : 하나의 상태. 가장 간단한 데이터 클래스
*env_set : env_state를 리스트로 가지는 연속된 환경과 act에 따른 env_state의 추가 및 변경
*Agent  : emv_set과 연동

하려 했으나 강화 학습은 결국 라이브러리를 쓰는게 맞는거 같기도....
"""