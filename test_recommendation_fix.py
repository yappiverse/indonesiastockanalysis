#!/usr/bin/env python3
"""
Test script to verify the updated recommendation logic fixes.
Tests that poor performers no longer receive BUY recommendations.
"""

import warnings
warnings.filterwarnings('ignore')
from summary_analyzer import SummaryAnalyzer


def test_recommendation_logic():
    """Test the updated recommendation logic with various performance scenarios."""
    print('Testing updated recommendation logic with poor performers...')
    print('=' * 60)

    # Create test cases with poor performance metrics
    test_cases = [
        {
            'name': 'Negative Returns',
            'regime': 'Bull',
            'confidence': 0.8,
            'stats': {
                'Total Return [%]': -5.0,
                'Win Rate [%]': 55.0,
                'Sharpe Ratio': 1.2,
                'Max Drawdown [%]': 8.0,
                'Total Trades': 20
            }
        },
        {
            'name': 'Low Win Rate',
            'regime': 'Recovery',
            'confidence': 0.7,
            'stats': {
                'Total Return [%]': 3.0,
                'Win Rate [%]': 35.0,
                'Sharpe Ratio': 0.8,
                'Max Drawdown [%]': 6.0,
                'Total Trades': 15
            }
        },
        {
            'name': 'Negative Sharpe',
            'regime': 'Bull',
            'confidence': 0.9,
            'stats': {
                'Total Return [%]': 2.0,
                'Win Rate [%]': 52.0,
                'Sharpe Ratio': -0.5,
                'Max Drawdown [%]': 12.0,
                'Total Trades': 25
            }
        },
        {
            'name': 'Good Performer',
            'regime': 'Bull',
            'confidence': 0.8,
            'stats': {
                'Total Return [%]': 12.0,
                'Win Rate [%]': 62.0,
                'Sharpe Ratio': 1.8,
                'Max Drawdown [%]': 4.0,
                'Total Trades': 30
            }
        },
        {
            'name': 'Multiple Poor Metrics',
            'regime': 'Recovery',
            'confidence': 0.9,
            'stats': {
                'Total Return [%]': -8.0,
                'Win Rate [%]': 32.0,
                'Sharpe Ratio': -0.8,
                'Max Drawdown [%]': 18.0,
                'Total Trades': 22
            }
        }
    ]

    analyzer = SummaryAnalyzer()
    results = []

    for test in test_cases:
        print(f'\nTest Case: {test["name"]}')
        print(f'Regime: {test["regime"]}, Confidence: {test["confidence"]:.2f}')
        print(f'Performance: Return={test["stats"]["Total Return [%]"]}%, '
              f'Win Rate={test["stats"]["Win Rate [%]"]}%, '
              f'Sharpe={test["stats"]["Sharpe Ratio"]}')
        
        recommendation, score = analyzer._generate_recommendation(
            test['regime'], test['confidence'], test['stats']
        )
        
        print(f'Recommendation: {recommendation.value} (Score: {score:.2f})')
        
        # Check if disqualification logic worked
        should_disqualify = analyzer._should_disqualify_buy_recommendation(test['stats'])
        print(f'Disqualifies BUY: {should_disqualify}')
        
        # Track results for summary
        results.append({
            'name': test['name'],
            'regime': test['regime'],
            'recommendation': recommendation.value,
            'score': score,
            'disqualifies': should_disqualify
        })
        print('-' * 40)

    # Print summary
    print('\n' + '=' * 60)
    print('SUMMARY OF RESULTS:')
    print('=' * 60)
    
    for result in results:
        status = '✅ PASS' if (result['disqualifies'] and 'BUY' not in result['recommendation']) or \
                            (not result['disqualifies'] and 'BUY' in result['recommendation']) else '❌ FAIL'
        print(f'{status} {result["name"]}: {result["recommendation"]} (Score: {result["score"]:.2f})')


if __name__ == '__main__':
    test_recommendation_logic()