(define (problem picture_57)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_sensor blue_regulator red_sensor - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_sensor)
        (clear blue_regulator)
        (clear red_sensor)
        (part_at blue_sensor table)
        (part_at blue_regulator table)
        (part_at red_sensor table)
    )
    
    (:goal
        (and
            (part_at blue_regulator regulator_placement)
        )
    )
)
