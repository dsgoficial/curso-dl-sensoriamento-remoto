// src/components/ExerciseCard/index.js
import React from 'react';
import styles from './styles.module.css';

export default function ExerciseCard({
  title,
  duration,
  difficulty,
  type,
  description,
  notebookUrl,
  solutionUrl
}) {
  const getDifficultyColor = (level) => {
    const colors = {
      'Iniciante': 'success',
      'Intermediário': 'warning', 
      'Avançado': 'danger',
      'Especialista': 'secondary'
    };
    return colors[level] || 'primary';
  };

  return (
    <div className={styles.exerciseCard}>
      <h3>{title}</h3>
      <div className={styles.meta}>
        <span className="badge badge--info">⏱️ {duration}</span>
        <span className={`badge badge--${getDifficultyColor(difficulty)}`}>
          📊 {difficulty}
        </span>
        <span className="badge badge--secondary">🔧 {type}</span>
      </div>
      <p>{description}</p>
      <div className={styles.actions}>
        <a 
          href={notebookUrl} 
          className="button button--primary"
          target="_blank"
          rel="noopener noreferrer"
        >
          📓 Abrir no Colab
        </a>
        {solutionUrl && (
          <a 
            href={solutionUrl} 
            className="button button--secondary button--outline"
          >
            ✅ Ver Solução
          </a>
        )}
      </div>
    </div>
  );
}