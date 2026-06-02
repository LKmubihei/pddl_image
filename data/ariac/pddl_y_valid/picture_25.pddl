(define (problem picture_25)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_sensor green_sensor green_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_sensor)
        (clear green_sensor)
        (clear green_regulator)
        (part_at red_sensor table)
        (part_at green_sensor table)
        (part_at green_regulator table)
    )
    
    (:goal
        (and
            (part_at green_regulator regulator_placement)
        )
    )
)
